import os
import torch
from torch.utils.data import Dataset
import numpy as np


def load_output(directory, item_id, prefix=''):
    file_path = os.path.join(directory, f"{prefix}_{item_id}.pt")

    if os.path.exists(file_path):
        return torch.load(file_path, weights_only=True)
    else:
        return None


class ModalityDataset(Dataset):
    def __init__(self, args, u2seq, query, k):
        assert 0 <= k <= 1
        self.u2seq = u2seq
        self.users = list(u2seq.keys())
        self.max_seq_len = args.max_seq_len + 1
        self.query = query
        self.modality = args.modality
        self.emb_size = args.word_embedding_dim
        self.stored_vector_path = args.stored_vector_path.format(args.dataset_name)
        self.side_adapter_num_list = [0] + [int(i) for i in list(args.layers.split(","))]

        # count freq
        item_freq = {}
        for seq in u2seq.values():
            for item in seq:
                item_freq[item] = item_freq.get(item, 0) + 1

        sorted_items = sorted(item_freq.items(), key=lambda x: x[1], reverse=True)
        topk_count = int(len(sorted_items) * k)
        self.top_popular_items = set(item for item, _ in sorted_items[:topk_count])

    def __len__(self):
        return len(self.u2seq)

    def process_seqs(self, idx):
        user_id = self.users[idx]
        seq = self.u2seq[user_id]
        tokens_len = len(seq) - 1
        # 11 - x >= 1?
        mask_len = self.max_seq_len - len(seq)
        # 11 - x  + x - 1 = 10
        log_mask = [0] * mask_len + [1] * tokens_len
        # length = 11 - x + x = 11
        item_ids = torch.LongTensor([0] * mask_len + seq)
        return seq, mask_len, item_ids, log_mask


class SingleModalDataset(ModalityDataset):
    def __init__(self, args, u2seq, query, modality, k=1.0):
        super().__init__(args, u2seq, query, k)
        self.pretrained_outputs = {}
        if modality == 'cv':
            self.prefix = 'vit'
            self.folder = '/vit_outputs'
        else:
            assert modality == 'text'
            self.prefix = 'bert'
            self.folder = '/bert_outputs'

        for item_id in self.top_popular_items:
            item_name = query.id_to_name(item_id, 'item')
            full_h_states = load_output(self.stored_vector_path + self.folder, item_name, prefix=self.prefix)
            self.pretrained_outputs[item_id] = full_h_states[self.side_adapter_num_list, :]

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, idx):
        seq, mask_len, item_ids, log_mask = super().process_seqs(idx)
        samples = []

        for i in range(len(seq)):
            item_id = seq[i]
            if item_id in self.top_popular_items:
                output_loaded = self.pretrained_outputs[item_id]
            else:
                item_name = self.query.id_to_name(item_id, 'item')
                full_h_states = load_output(self.stored_vector_path + self.folder, item_name, prefix=self.prefix)
                output_loaded = full_h_states[self.side_adapter_num_list, :]
            samples.append(output_loaded)
        samples = [torch.zeros(samples[0].shape)] * mask_len + samples
        samples = torch.stack(samples)
        return item_ids, samples, torch.FloatTensor(log_mask)


def compute_input_output_shifts(outputs_dict, layer_indices):
    """
    Computes average L2 norm between input and output of each layer.

    Args:
        outputs_dict (dict): item_id -> tensor of shape (num_layers + 1, hidden_dim)
                             includes h_0 (input) through h_L (final layer output)
        layer_indices (list): list of layer indices (e.g., [1, 2, 3, ...])

    Returns:
        dict: layer_index -> average L2 shift across all items
    """
    shift_sums = {l: 0.0 for l in layer_indices}
    shift_counts = {l: 0 for l in layer_indices}

    for item_id, h_states in outputs_dict.items():
        for i, l in enumerate(layer_indices):
            if l == 0:
                input_to_layer = torch.zeros_like(h_states[l])
            else:
                input_to_layer = h_states[l - 1]  # h_{l-1}
            output_of_layer = h_states[l]     # h_l
            shift = torch.norm(output_of_layer - input_to_layer, p=2).item()
            shift_sums[l] += shift
            shift_counts[l] += 1

    avg_shifts = {l: shift_sums[l] / shift_counts[l] for l in layer_indices if shift_counts[l] > 0}
    return avg_shifts



class MMDataset(ModalityDataset):
    def __init__(self, args, u2seq, query, k=1.0):
        super().__init__(args, u2seq, query, k)
        self.vit_outputs = {}
        self.bert_outputs = {}

        for item_id in self.top_popular_items:
            item_name = query.id_to_name(item_id, 'item')
            full_h_states_vit = load_output(self.stored_vector_path + '/vit_outputs', item_name, prefix='vit')
            full_h_states_bert = load_output(self.stored_vector_path + '/bert_outputs', item_name, prefix='bert')
            self.vit_outputs[item_id] = full_h_states_vit[self.side_adapter_num_list, :]
            self.bert_outputs[item_id] = full_h_states_bert[self.side_adapter_num_list, :]

    def __getitem__(self, idx):
        seq, mask_len, item_ids, log_mask = super().process_seqs(idx)

        sample_texts = []
        sample_images = []

        for i in range(len(seq)):
            item_id = seq[i]
            if item_id in self.top_popular_items:
                bert_output_loaded = self.bert_outputs[item_id]
                vit_output_loaded = self.vit_outputs[item_id]
            else:
                item_name = self.query.id_to_name(item_id, 'item')
                full_h_states_vit = load_output(self.stored_vector_path + '/vit_outputs', item_name, prefix='vit')
                full_h_states_bert = load_output(self.stored_vector_path + '/bert_outputs', item_name, prefix='bert')
                bert_output_loaded = full_h_states_bert[self.side_adapter_num_list, :]
                vit_output_loaded = full_h_states_vit[self.side_adapter_num_list, :]
            sample_texts.append(bert_output_loaded)
            sample_images.append(vit_output_loaded)
        sample_texts = [torch.zeros(sample_texts[0].shape)] * mask_len + sample_texts
        sample_images = [torch.zeros(sample_images[0].shape)] * mask_len + sample_images
        sample_texts = torch.stack(sample_texts)
        sample_images = torch.stack(sample_images)
        return item_ids, sample_images, sample_texts, torch.FloatTensor(log_mask)


class ModalityEvalDatasetCached(Dataset):
    def __init__(self, args, item_set, query):
        self.data = item_set
        self.query = query
        self.stored_vector_path = args.stored_vector_path
        self.side_adapter_num_list = [0] + [int(i) for i in list(args.layers.split(","))]
        self.modality = args.modality

    def __len__(self):
        return len(self.data)


class SingleModalEvalDatasetCached(ModalityEvalDatasetCached):
    def __init__(self, args, item_set, query, outputs, modality):
        super().__init__(args, item_set, query)
        self.pretrained_outputs = outputs
        self.modality = modality
        if modality == 'cv':
            self.prefix = 'vit'
            self.folder = '/vit_outputs'
        else:
            assert modality == 'text'
            self.prefix = 'bert'
            self.folder = '/bert_outputs'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # pos
        item_id = self.data[index]

        if item_id != 0:
            item_name = self.query.id_to_name(item_id, 'item')
        else:
            num_trans = len(self.side_adapter_num_list)
            return torch.zeros((num_trans, 768))

        if item_id in self.pretrained_outputs:
            output_loaded = self.pretrained_outputs[item_id]
        else:
            full_h_states = load_output(self.stored_vector_path + self.folder, item_name, prefix=self.prefix)
            output_loaded = full_h_states[self.side_adapter_num_list, :]
        return output_loaded


class MMEvalDatasetCached(ModalityEvalDatasetCached):
    def __init__(self, args, item_set, query, vit_outputs, bert_outputs):
        super().__init__(args, item_set, query)
        self.vit_outputs = vit_outputs
        self.bert_outputs = bert_outputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # pos
        item_id = self.data[index]

        if item_id != 0:
            item_name = self.query.id_to_name(item_id, 'item')
        else:
            num_trans = len(self.side_adapter_num_list)
            return torch.zeros((num_trans, 768)), torch.zeros((num_trans, 768))

        if item_id in self.bert_outputs and item_id in self.vit_outputs:
            bert_output_loaded = self.bert_outputs[item_id]
            vit_output_loaded = self.vit_outputs[item_id]
        else:
            full_h_states_vit = load_output(self.stored_vector_path + '/vit_outputs', item_name, prefix='vit')
            full_h_states_bert = load_output(self.stored_vector_path + '/bert_outputs', item_name, prefix='bert')
            bert_output_loaded = full_h_states_bert[self.side_adapter_num_list, :]
            vit_output_loaded = full_h_states_vit[self.side_adapter_num_list, :]
        return vit_output_loaded, bert_output_loaded


class MMEvalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item2idx):
        self.u2seq = u2seq
        self.users = list(self.u2seq.keys())
        self.max_seq_len = max_seq_len + 1
        # Build local item index map: item2idx = { item_id: local_idx, ... }
        self.item2idx = item2idx
        # from get mm
        self.item_content = item_content

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        seq = self.u2seq[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        if target == 0:
            raise ValueError('target cannot equal 0!')
        mask_len = self.max_seq_len - len(seq)

        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)

        # Convert pad_tokens from global item ID to local index
        #  For 0, we can map to local idx 0 as dummy or skip
        # But let's do a safe lookup:
        pad_indices = []
        for itm in pad_tokens:
            if itm in self.item2idx:
                pad_indices.append(self.item2idx[itm])
            else:
                pad_indices.append(0)  # or handle differently

        # shape => (len_seq, 13, 768)
        input_embs = self.item_content[pad_indices]

        # Construct label vector of size len(item_set)
        labels = np.zeros(len(self.item2idx) - 1, dtype=np.float32)
        labels[self.item2idx[target] - 1] = 1.0

        return (
            torch.LongTensor([user_id]),
            input_embs,
            torch.FloatTensor(log_mask),
            torch.from_numpy(labels)
        )
