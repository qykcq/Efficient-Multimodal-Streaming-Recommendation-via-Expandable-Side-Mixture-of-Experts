from collections import defaultdict
import torch
import numpy as np
import time
from model.recommender import ModelMM
from metrics import get_mm_item_embs, eval_model
from preprocess import get_train_valid_test_sets, read_json_to_sorted_kcore
import functools
import os
import random


def monitor_gpu_memory(func):
    """
    Decorator that prints GPU memory stats when the wrapped function finishes.
    Reports:
      - current allocated
      - current reserved (cached by CUDA allocator)
      - peak allocated during the call
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # make sure CUDA is in a clean state
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # run user code
        result = func(*args, **kwargs)
        # only report if CUDA is available
        if torch.cuda.is_available():
            alloc      = torch.cuda.memory_allocated()
            reserved   = torch.cuda.memory_reserved()
            peak_alloc = torch.cuda.max_memory_allocated()
            print(f"[GPU] allocated:   {alloc/1024**2:7.2f} MB")
            print(f"[GPU] reserved:    {reserved/1024**2:7.2f} MB")
            print(f"[GPU] peak alloc:  {peak_alloc/1024**2:7.2f} MB")
        return result
    return wrapper


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_item2idx(users_valid):
    val_item2idx = {0: 0}
    count = 1
    for values in users_valid.values():
        for item in values:
            if item not in val_item2idx:
                val_item2idx[item] = count
                count += 1
    return val_item2idx


def training_setup(users_valid, users_test, hist_for_valid):
    max_epoch, max_eval_ndcg, early_stop_count = 1, 0, 0
    best_ckp = None  # Store the best model state
    val_item2idx = get_item2idx(users_valid)
    filtered_hist_for_valid = defaultdict(list)
    for user in hist_for_valid:
        for item in hist_for_valid[user]:
            if item in val_item2idx:
                filtered_hist_for_valid[user].append(val_item2idx[item])
        hist_for_valid[user] = None

    item_num_test = max([max(users_test[user]) for user in users_test])
    test_item2idx = {i: i for i in range(item_num_test + 1)}
    # item2idx start from 0 (nonexistent) and end with the actual largest item index.
    # therefore, len(item2idx) = item_num + 1 because it has 0
    return max_epoch, max_eval_ndcg, early_stop_count, best_ckp, filtered_hist_for_valid, test_item2idx, val_item2idx


class Engine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = args.batch_size
        set_seed(args.seed)

        # training related
        self.overall_perf = {}
        self.timestamp = 1
        self.query = None
        self.model = None

        self.time_counter = 0

    def print_final_performance(self):
        print('%' * 50 + 'Final Performance Report' + '%' * 50)
        metric_sum = [0, 0, 0, 0]
        for t in self.overall_perf:
            print('{}: HIT@10 {:.2f}, NDCG@10 {:.2f}, Hit@20 {:.2f}, NDCG@20 {:.2f}'.format(
                t, self.overall_perf[t][0], self.overall_perf[t][1], self.overall_perf[t][2], self.overall_perf[t][3]))
            metric_sum[0] += self.overall_perf[t][0]
            metric_sum[1] += self.overall_perf[t][1]
            metric_sum[2] += self.overall_perf[t][2]
            metric_sum[3] += self.overall_perf[t][3]
        num = len(self.overall_perf)
        print('AVG HIT@10 {:.2f} NDCG@10 {:.2f} HIT@20 {:.2f} NDCG@20 {:.2f}'.format(
            metric_sum[0] / num, metric_sum[1] / num, metric_sum[2] / num, metric_sum[3] / num
        ))

    def get_parameter_stats(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Assuming 4 bytes (float32) per parameter
        total_size_mb = total_params * 4 / (1024 ** 2)
        trainable_size_mb = trainable_params * 4 / (1024 ** 2)

        percent = 100 * trainable_params / total_params if total_params > 0 else 0

        print(f"Total Parameters:      {total_params:,}")
        print(f"Trainable Parameters:  {trainable_params:,}")
        print(f"Trainable %:           {percent:.2f}%")
        print(f"Model Size:            {total_size_mb:.2f} MB (float32)")
        print(f"Trainable Parameter Size: {trainable_size_mb:.2f} MB (float32)")

    # @monitor_gpu_memory
    # def init_train(self, dfs):
    #     self.timestamp = 1
    #     print('#' * 50 + f'Time Window 1' + '#' * 50, flush=True)
    #     train_dl, pop_prob_list, users_valid, users_test, hist_for_valid, \
    #         hist_for_test = get_train_valid_test_sets_replay(self.args, self.query, dfs, 0)
    #     # train till convergence
    #     best_ckpt, hist_test, users_test, test_item2idx, train_dl = \
    #         self.update_modules(train_dl, pop_prob_list, users_valid, users_test, hist_for_valid, hist_for_test)
    #
    #     self.test(best_ckpt, hist_test, users_test, test_item2idx, train_dl)
    #
    #     self.get_parameter_stats()
    #     # if self.args.biastune:
    #     #     self.model.multimodal_encoder.freeze_experts()

    @monitor_gpu_memory
    def train(self):
        self.model = ModelMM(self.args).to(self.device)
        self.get_parameter_stats()
        dfs, self.query = read_json_to_sorted_kcore(self.args)

        for t in range(1, self.args.time_windows):
            self.timestamp = t
            print('#' * 50 + f'Time Window {t}' + '#' * 50, flush=True)
            train_dl, pop_prob_list, users_valid, users_test, hist_for_valid, \
                hist_for_test = get_train_valid_test_sets(self.args, self.query, dfs, t - 1)

            # train till convergence
            best_ckpt, hist_test, users_test, test_item2idx, train_dl = \
                self.update_modules(train_dl, pop_prob_list, users_valid, users_test, hist_for_valid, hist_for_test)

            print(f'Training time for the {t}th window', self.time_counter)
            self.time_counter = 0

            self.test(best_ckpt, hist_test, users_test, test_item2idx, train_dl)

            self.get_parameter_stats()

            for encoder in self.model.multimodal_encoder.get_encoders():
                encoder.prune()
                encoder.expand()

            self.model.update_optimizer()

        self.print_final_performance()

    def validate(self, ep, best_checkpoint, max_epoch, max_eval_ndcg, early_stop_count,
                 hist_for_valid, users_valid, val_item2idx, train_dl):
        if ep + 1 <= self.args.warmup_epochs:
            return False, False, None, max_eval_ndcg, early_stop_count, max_epoch

        self.model.eval()
        if self.args.modality in ['mm']:
            new_best_found, max_eval_ndcg, max_epoch, early_stop_count, need_break = self.run_eval(
                ep + 1, max_epoch, max_eval_ndcg, early_stop_count, hist_for_valid, users_valid, val_item2idx,
                vit_outputs=train_dl.dataset.vit_outputs, bert_outputs=train_dl.dataset.bert_outputs)
        elif self.args.modality == 'cv':
            new_best_found, max_eval_ndcg, max_epoch, early_stop_count, need_break = self.run_eval(
                ep + 1, max_epoch, max_eval_ndcg, early_stop_count, hist_for_valid, users_valid, val_item2idx,
                vit_outputs=train_dl.dataset.pretrained_outputs, bert_outputs=None)
        else:
            new_best_found, max_eval_ndcg, max_epoch, early_stop_count, need_break = self.run_eval(
                ep + 1, max_epoch, max_eval_ndcg, early_stop_count, hist_for_valid, users_valid, val_item2idx,
                vit_outputs=None, bert_outputs=train_dl.dataset.pretrained_outputs)

        # Save model checkpoint if new best validation performance is found
        if new_best_found:
            best_checkpoint = self.model.state_dict()  # Store model state
            print(f"New best validation NDCG: {max_eval_ndcg:.5f} (Epoch {ep + 1}). Checkpoint saved.")

        return new_best_found, need_break, best_checkpoint, max_eval_ndcg, early_stop_count, max_epoch

    def test(self, best_checkpoint, hist_for_test, users_test, item2idx, train_dl):

        # Load the best model checkpoint before testing
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint)

        # Perform test only after training completes
        self.model.eval()

        if self.args.modality in ['mm']:
            metrics = self.run_test(hist_for_test, users_test, item2idx,
                vit_outputs=train_dl.dataset.vit_outputs, bert_outputs=train_dl.dataset.bert_outputs)
        elif self.args.modality == 'cv':
            metrics = self.run_test(hist_for_test, users_test, item2idx,
                vit_outputs=train_dl.dataset.pretrained_outputs, bert_outputs=None)
        else:
            assert self.args.modality == 'text'
            metrics = self.run_test(hist_for_test, users_test, item2idx,
                vit_outputs=None, bert_outputs=train_dl.dataset.pretrained_outputs)

        self.overall_perf[self.timestamp] = [m.item() for m in metrics]

    def update_modules(self, train_dl, pop_prob_list, users_valid, users_test, hist_valid, hist_test):
        max_epoch, max_eval_ndcg, early_stop_count, best_ckpt, hist_valid, \
            test_item2idx, val_item2idx = training_setup(users_valid, users_test, hist_valid)
        for ep in range(self.args.epochs):
            print('-' * 25 + 'Time Window {}, Epoch {}'.format(self.timestamp, ep + 1) + '-' * 25, flush=True)

            self.time_counter += self.model.update_one_epoch(train_dl, pop_prob_list)
            new_best_found, need_break, best_ckpt, max_eval_ndcg, early_stop_count, max_epoch = self.validate(
                ep, best_ckpt, max_epoch, max_eval_ndcg, early_stop_count,
                hist_valid, users_valid, val_item2idx, train_dl)

            if need_break:
                break

        return best_ckpt, hist_test, users_test, test_item2idx, train_dl

    def run_test(self, hist_for_test, users_test, item2idx, vit_outputs=None, bert_outputs=None):
        t1 = time.time()
        item_embs = get_mm_item_embs(self.args, self.model, item2idx, self.query,
                                     self.args.batch_size, vit_outputs, bert_outputs)
        h10, n10, h20, n20 = eval_model(self.model, hist_for_test, users_test, item_embs, self.args, item2idx, 'Test')
        t2 = time.time()
        print("{:.2f}s spent for test".format(t2 - t1), flush=True)
        return h10, n10, h20, n20

    def run_eval(self, ep, max_epoch, max_eval_value, early_stop_count, hist, users_eval,
                 item2idx, vit_outputs=None, bert_outputs=None):
        t1 = time.time()
        item_embs = get_mm_item_embs(self.args, self.model, item2idx, self.query,
                                     self.args.batch_size, vit_outputs,bert_outputs)
        valid_hit10, valid_ndcg, _, _ = eval_model(self.model, hist, users_eval, item_embs, self.args, item2idx, 'Eval')
        need_break = False
        new_best_found = False
        if valid_ndcg > max_eval_value:
            max_eval_value = valid_ndcg
            max_epoch = ep
            early_stop_count = 0
            new_best_found = True
        else:
            early_stop_count += 1
            if early_stop_count >= self.args.patience:
                need_break = True
        print("{:.2f}s spent for validation, early stop count {}".format(time.time() - t1, early_stop_count), flush=True)

        return new_best_found, max_eval_value, max_epoch, early_stop_count, need_break


def save_checkpoint(model: torch.nn.Module,
                    epoch: int,
                    path: str):
    """
    Saves model & optimizer state plus epoch counter to `path`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': model.optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model: torch.nn.Module,
                    path: str,
                    device: torch.device):
    """
    Loads model & optimizer state from `path` into the given instances,
    and returns the last saved epoch.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.optimizer.load_state_dict(checkpoint['optim_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint '{path}' (resuming from epoch {start_epoch})")
    return start_epoch



