import time
import torch
import torch.optim as optim
from torch import nn
from mm_encoders import XSMoEMMEncoder
from mm_modules import UserEncoder


class ModelMM(torch.nn.Module):
    def __init__(self, args):
        super(ModelMM, self).__init__()
        self.args = args
        # max number of items per user
        self.max_seq_len = args.max_seq_len
        self.num_h_states = 1 + len(args.layers.split(','))

        # Encodes user-item iteraction sequences
        self.user_encoder = UserEncoder(args)
        self.criterion = nn.CrossEntropyLoss()

        self.multimodal_encoder = XSMoEMMEncoder(args)

        self.scaler = torch.amp.GradScaler('cuda')

        trainable_params = []
        for name, params in self.named_parameters():
            if params.requires_grad:
                trainable_params.append(params)
        self.optimizer = optim.Adam([
            {'params': trainable_params, 'lr': self.args.lr}
        ])

    def calculate_ce_loss(self, ids, log_mask, prec_vec, score_embs, pop_prob_list):
        device = next(self.parameters()).device
        # Debiasing: converts pop_prob_list to logarithmic scale and adjusts logits
        pop_prob_list = torch.FloatTensor(pop_prob_list).to(device)
        debias_logits = torch.log(pop_prob_list[ids])

        # IN-BATCH CROSS-ENTROPY LOSS
        bs = log_mask.size(0)  # batch size

        # (bs * max_seq_len) => 1280
        ce_label = torch.tensor(
            [i * self.max_seq_len + i + j for i in range(bs) for j in range(1, self.max_seq_len + 1)],
            dtype=torch.long).to(device)
        # (bs * max_seq_len, bs * (max_seq_len + 1)) => (num users, num items + 1) => (1280, 1408)
        logits = torch.matmul(prec_vec, score_embs.t())
        if torch.isinf(logits).any():
            raise ValueError('inf values in logits!')
        if torch.isnan(logits).any():
            raise ValueError('nan values in logits!')

        # Debias with popularity
        logits = logits - debias_logits

        # Set the logits of invalid positions to 0
        logits[:, torch.cat((log_mask, torch.ones(log_mask.size(0))
                             .unsqueeze(-1).to(device)), dim=1).view(-1) == 0] = -1e4

        logits = logits.view(bs, self.max_seq_len, -1)

        id_list = ids.view(bs, -1)  # sample_items_id (bs, max_seq_len)
        """
            Mask out items that the user has already interacted with from the recommendation list.
            This ensures the model does not recommend items already seen by the user.
        """
        for i in range(bs):
            # it has shape of (max_seq_len) because each user interacts with up tp max_seq_len items
            reject_list = id_list[i]
            u_ids = ids.repeat(self.max_seq_len).expand((len(reject_list), -1))
            reject_mat = reject_list.expand((u_ids.size(1), len(reject_list))).t()
            # (max_seq_len, batch_size * (max_seq_len + 1))
            mask_mat = (u_ids == reject_mat).any(axis=0).reshape(logits[i].shape)
            for j in range(self.max_seq_len):
                mask_mat[j][i * (self.max_seq_len + 1) + j + 1] = False
            logits[i][mask_mat] = -1e4

        indices = torch.where(log_mask.view(-1) != 0)
        logits = logits.view(bs * self.max_seq_len, -1)

        loss = self.criterion(logits[indices], ce_label[indices])
        return loss

    def compute_scores(self, item_mm_seqs, log_mask, cand_item_mm_embs):
        device = next(self.parameters()).device
        cand_item_mm_embs = cand_item_mm_embs.to(device)
        prec_emb = self.user_encoder(item_mm_seqs, log_mask)[:, -1].detach()
        scores = torch.matmul(prec_emb, cand_item_mm_embs.t()).squeeze(dim=-1).detach()
        return scores

    def forward(self, ids, images, texts, log_mask, pop_prob_list):
        if images is not None:
            images = images.view(-1, 1 + self.max_seq_len, self.num_h_states, self.args.word_embedding_dim)
        if texts is not None:
            texts = texts.view(-1, 1 + self.max_seq_len, self.num_h_states, self.args.word_embedding_dim)
        ids = ids.view(-1)
        # mutimodality encoding
        score_embs, load_balancing_loss = self.multimodal_encoder(images, texts)

        input_embs = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)

        # we need to exclude the item to be predicted from the input to the user encoder
        prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask)

        # user preference vector where each row represents a user (1280, 128)
        prec_vec = prec_vec.reshape(-1, self.args.embedding_dim)  # (bs * max_seq_len, emb dim)

        ce_loss = self.calculate_ce_loss(ids, log_mask, prec_vec, score_embs, pop_prob_list)
        return ce_loss + load_balancing_loss

    def update_one_epoch(self, train_dl, pop_prob_list):
        if self.args.model == 'xsmoe':
            for encoder in self.multimodal_encoder.get_encoders():
                encoder.clear_tracker(0, list(range(encoder.num_layers)), True)

        self.train()
        device = next(self.parameters()).device
        total_loss = 0
        epoch_start_time = time.time()
        for batch in train_dl:
            if self.args.modality in ['mm']:
                ids, images, texts, masks = [x.to(device) for x in batch]
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss = self.forward(ids, images, texts, masks, pop_prob_list)
                    total_loss += loss
            elif self.args.modality == 'cv':
                ids, images, masks = [x.to(device) for x in batch]
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss = self.forward(ids, images, None, masks, pop_prob_list)
                    total_loss += loss
            else:
                ids, texts, masks = [x.to(device) for x in batch]
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss = self.forward(ids, None, texts, masks, pop_prob_list)
                    total_loss += loss
            # update sidenets and routers
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        delta_time = time.time() - epoch_start_time

        # update learning rate
        lrs = []
        for i in range(len(self.optimizer.param_groups)):
            current_lr = self.optimizer.param_groups[i]['lr']
            self.optimizer.param_groups[i]['lr'] = max(self.args.min_lr, current_lr * self.args.gamma)
            lr = self.optimizer.param_groups[i]['lr']
            lrs.append(round(lr, 4))
        print(f'Sum loss {total_loss:.4f}, {delta_time:.2f}s, LRs {lrs}', flush=True)
        if self.args.model == 'xsmoe':
            self.multimodal_encoder.print_util()
        return delta_time

    def update_optimizer(self):
        # current_lr = self.optimizer.param_groups[0]['lr']
        trainable_params = []
        for name, params in self.named_parameters():
            if params.requires_grad:
                trainable_params.append(params)
        self.optimizer = optim.Adam([
            {'params': trainable_params, 'lr': self.args.lr}
        ])
