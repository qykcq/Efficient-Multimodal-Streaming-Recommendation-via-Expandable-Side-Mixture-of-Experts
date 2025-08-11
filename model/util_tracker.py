import torch


class GateUtilizationTracker:
    def __init__(self, num_experts, num_layers):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.per_sample_norms = None
        # running sum of L2 norms, one per expert
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sum_norms = [torch.zeros(self.num_experts[i]).to(self.device) for i in range(num_layers)]
        self.sum_weights = [torch.zeros(self.num_experts[i]).to(self.device) for i in range(num_layers)]
        # total number of “samples” (i.e. rows in weighted_outs) seen
        self.total_samples = 0

    def reset(self, expert_change, layer, clear_sample_count):
        self.num_experts[layer] += expert_change
        self.sum_norms[layer] = torch.zeros(self.num_experts[layer]).to(self.device)
        self.sum_weights[layer] = torch.zeros(self.num_experts[layer]).to(self.device)
        if clear_sample_count:
            self.total_samples = 0

    @torch.no_grad()
    def update(self, weighted_outs, weights, layer):
        """
        weighted_outs: (B, E, D) — the expert outputs *after* applying the gate weights,
                                        for one batch.
        weights: (B, E) - the gate weights
        """
        # 1) compute per‐sample, per‐expert norms: (B, E)
        per_sample = weighted_outs.norm(p=2, dim=2)

        # 2) sum across the batch → (E,)
        batch_sum = per_sample.sum(dim=0)

        # 3) accumulate
        self.sum_norms[layer] += batch_sum
        self.sum_weights[layer] += weights.sum(0)
        self.total_samples += per_sample.size(0)

    @torch.no_grad()
    def utilization(self, layer) -> torch.Tensor:
        if self.total_samples == 0:
            assert False
            # return torch.zeros_like(self.sum_norms[layer])
        vals = self.sum_norms[layer] / self.total_samples
        total = vals.sum()
        vals = vals / total
        return vals

    @torch.no_grad()
    def gate_utilization(self, layer):
        if self.total_samples == 0:
            return torch.zeros_like(self.sum_weights[layer])
        vals = self.sum_weights[layer] / self.total_samples
        total = vals.sum()
        vals = vals / total
        return vals