import torch
from util_tracker import GateUtilizationTracker
from torch import nn
import torch.nn.functional as F


class MixtureExpert(nn.Module):
    def __init__(self, args, modality):
        super(MixtureExpert, self).__init__()
        self.args = args
        self.modality = modality
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experts = nn.ModuleList([Expert(args) for _ in range(args.num_experts)])

    def forward(self, x):
        outs = []
        for expert in self.experts:
            outs.append(expert(x))
        return outs

    def num_experts(self):
        return len(self.experts)

    def __freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def increment(self):
        with torch.no_grad():
            self.__freeze()
            # Create a new expert
            new_expert = Expert(self.args).to(self.device)

            # Get all previous expert parameters (assumes same structure)
            prev_params = [dict(expert.named_parameters()) for expert in self.experts]
            new_params = dict(new_expert.named_parameters())

            # Average each parameter
            for name in new_params:
                stacked = torch.stack([params[name].data for params in prev_params])
                mean_param = stacked.mean(dim=0)
                new_params[name].copy_(mean_param)

            self.experts.append(new_expert)
        print(self.modality, [any(p.requires_grad for p in expert.parameters()) for expert in self.experts])


class Expert(torch.nn.Module):
    def __init__(self, args):
        super(Expert, self).__init__()
        self.fc_down = nn.Linear(args.word_embedding_dim, args.adapter_down_size, bias=False)
        nn.init.normal_(self.fc_down.weight, std=1e-2)
        self.activate = nn.GELU()
        self.fc_up = nn.Linear(args.adapter_down_size, args.word_embedding_dim, bias=False)
        nn.init.normal_(self.fc_up.weight, std=1e-2)

    def forward(self, input_embs):
        # the original input embs + the output of an autoencoder
        x = self.fc_down(input_embs)
        x = self.activate(x)
        return self.fc_up(x) + input_embs


class Router(nn.Module):
    def __init__(self, args):
        super(Router, self).__init__()
        self.linear = nn.Linear(args.word_embedding_dim, args.num_experts + 1, bias=False)

    def forward(self, x):
        output_x = self.linear(x)
        weights = F.softmax(output_x, dim=-1)
        return weights, 0

    def increment(self):
        device = next(self.parameters()).device
        in_features = self.linear.in_features
        out_features = self.linear.out_features

        # Create new linear layer with one additional output dimension and no bias
        new_linear = nn.Linear(in_features, out_features + 1, bias=False).to(device)

        # Copy old weights
        with torch.no_grad():
            new_linear.weight[:out_features] = self.linear.weight

        # Replace the old linear layer
        self.linear = new_linear


class LadderSMoE(nn.Module):
    def __init__(self, args, modality):
        super(LadderSMoE, self).__init__()
        self.args = args
        self.expert = Expert
        self.modality = modality
        num_layers = len(args.layers.split(','))
        self.num_layers = num_layers
        # n layers of mixture of experts
        self.moes = nn.ModuleList([MixtureExpert(args, modality) for _ in range(num_layers)])
        self.routers = nn.ModuleList([
            Router(args) for _ in range(num_layers)
        ])

        num_experts_per_layer = [self.moes[i].num_experts() + 1 for i in range(len(self.moes))]
        self.tracker = GateUtilizationTracker(num_experts_per_layer, num_layers)

    def forward(self, mod):
        lb_losses = 0
        x = mod[..., 0, :].reshape(-1, self.args.word_embedding_dim)
        for i in range(len(self.moes)):
            expert_out = [mod[..., i + 1, :].reshape(-1, self.args.word_embedding_dim)]
            expert_out.extend(self.moes[i](x))  # Apply expert
            router_out, lb_loss = self.routers[i](x)  # Get router weights
            stacked_outs = torch.stack(expert_out, dim=1)
            weighted_outs = router_out.unsqueeze(-1) * stacked_outs
            x = weighted_outs.sum(dim=1) # Weighted sum
            lb_losses += lb_loss

            self.tracker.update(weighted_outs, router_out, i)
        return x, lb_losses

    def clear_tracker(self, expert_change, layers, clear_sample_count):
        for layer in layers:
            self.tracker.reset(expert_change, layer, clear_sample_count)

    def expand(self):
        with torch.no_grad():
            for mixture_expert in self.moes:
                mixture_expert.increment()

            for router in self.routers:
                router.increment()
            # expansion occurs on all layers
            self.clear_tracker(1, list(range(self.num_layers)), False)

    def prune(self):
        thresh = self.args.tau

        for layer in range(self.num_layers):
            # utilities for [base, expert1, expert2, …]
            utils = self.tracker.utilization(layer)
            print(self.modality, layer, utils.tolist())
            # ignore base at index 0
            expert_utils = utils[1:] / torch.sum(utils[1:])
            # find the worst expert
            min_util, rel_idx = min((u, i) for i, u in enumerate(expert_utils))

            if min_util >= thresh:
                continue  # nothing to prune in this layer
            print('Modality {}: min util {:.2f} pruning expert {} at layer {}'.format(
                self.modality, min_util, rel_idx + 1, layer))
            self.clear_tracker(-1, [layer], False)

            # absolute index in the weight matrix
            prune_idx = rel_idx + 1

            comp_expert = self.moes[layer]
            router = self.routers[layer]
            device = next(router.parameters()).device

            # remove the Expert module (ModuleList index = rel_idx)
            del comp_expert.experts[rel_idx]

            # rebuild router.linear without that row
            old_w   = router.linear.weight.data
            in_f    = router.linear.in_features
            old_out = router.linear.out_features
            new_out = old_out - 1

            new_lin = nn.Linear(in_f, new_out, bias=False).to(device)
            with torch.no_grad():
                # copy rows before and after prune_idx
                new_lin.weight.data[:] = torch.cat([
                    old_w[:prune_idx],
                    old_w[prune_idx + 1:]
                ], dim=0)

            router.linear      = new_lin

