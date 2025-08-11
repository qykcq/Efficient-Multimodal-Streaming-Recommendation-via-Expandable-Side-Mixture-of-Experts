from torch.utils.data import DataLoader
from dataset import MMEvalDatasetCached, MMEvalDataset, SingleModalEvalDatasetCached
import torch
import math
import numpy as np


def collate_mm(batch):

    imgs, texts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    texts = torch.stack(texts, dim=0)
    return imgs, texts

def collate_single_modality(batch):
    content = torch.stack(batch, dim=0)
    return content


def get_mm_item_embs(args, model, item2idx, query, test_batch_size, vit_outputs=None, bert_outputs=None):
    model.eval()
    device = next(model.parameters()).device
    if args.modality in ['mm']:
        item_dataset = MMEvalDatasetCached(args, list(item2idx.keys()), query, vit_outputs, bert_outputs)
        my_collate = collate_mm
    elif args.modality == 'cv':
        item_dataset = SingleModalEvalDatasetCached(args, list(item2idx.keys()), query, vit_outputs, 'cv')
        my_collate = collate_single_modality
    else:
        item_dataset = SingleModalEvalDatasetCached(args, list(item2idx.keys()), query, bert_outputs, 'text')
        my_collate = collate_single_modality

    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, collate_fn=my_collate, shuffle=False)
    item_embeddings = []
    with torch.no_grad():
        for data_batch in item_dataloader:
            if args.modality in ['mm']:
                input_ids_cv, input_ids_text = [x.to(device) for x in data_batch]

            elif args.modality == 'cv':
                input_ids_cv = data_batch.to(device)
                input_ids_text = None
            else:
                assert args.modality == 'text'
                input_ids_cv = None
                input_ids_text = data_batch.to(device)
            concat_emb, _ = model.multimodal_encoder(input_ids_cv, input_ids_text)
            concat_emb = concat_emb.to(torch.device('cpu')).detach()
            item_embeddings.extend(concat_emb)
    return torch.stack(tensors=item_embeddings, dim=0)


def eval_model(model, user_history, eval_seq, concat_embs, args, item2idx, v_or_t):
    device = next(model.parameters()).device

    eval_dataset = MMEvalDataset(eval_seq, concat_embs, args.max_seq_len, item2idx)
    eval_dl = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        eval_all_user = []
        # item rank does not include 0
        # item num = len(item2idx) - 1
        item_rank = torch.tensor(np.arange(1, len(item2idx))).to(device)
        for batch in eval_dl:
            user_ids, item_mm_embs, log_mask, labels = [x.to(device) for x in batch]
            # the scores contain score for the nonexistent item 0
            scores = model.compute_scores(item_mm_embs, log_mask, concat_embs)

            # the score on the first column will be very small
            for user_id, label, score in zip(user_ids, labels, scores):
                # the label here is already converted to local index
                user_id = user_id[0].item()
                if user_id in user_history:
                    history = user_history[user_id]
                    score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topk(score, label, item_rank, device))

        Hit10, nDCG10, Hit20, nDCG20 = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        mean_eval = eval_concat([Hit10 * 100, nDCG10 * 100, Hit20 * 100, nDCG20 * 100])
        print(f"{v_or_t} Hit10 = {mean_eval[0]}, nDCG10 = {mean_eval[1]}", flush=True)
        print(f"{v_or_t} Hit20 = {mean_eval[2]}, nDCG20 = {mean_eval[3]}", flush=True)
    return mean_eval


def eval_concat(eval_list):
    eval_result = []
    for eval_m in eval_list:
        eval_m_cpu = eval_m.to(torch.device("cpu")).numpy()
        eval_result.append(eval_m_cpu.mean())
    return eval_result


def metrics_topk(y_score, y_true, item_rank, local_rank):
    # predicted rank
    order = torch.argsort(y_score, descending=True)
    y_true = torch.take(y_true, order)

    rank = torch.sum(y_true * item_rank)
    eval_ra = torch.zeros(4).to(local_rank)
    if rank <= 10:
        eval_ra[0] = 1
        eval_ra[1] = 1 / math.log2(rank + 1)
    if rank <= 20:
        eval_ra[2] = 1
        eval_ra[3] = 1 / math.log2(rank + 1)

    return eval_ra



