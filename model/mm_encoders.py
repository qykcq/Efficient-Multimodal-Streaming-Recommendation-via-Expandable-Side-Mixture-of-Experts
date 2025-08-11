from torch import nn
from experts import LadderSMoE
import torch


def print_structure(modalities):
    def check_frozen_status(params):
        """Helper function to check if a parameter (or module) is frozen (requires_grad=False)."""
        return all(not p.requires_grad for p in params)

    print("==== Model Structure ====")
    for modality, expert_obj in modalities.items():
        num_experts = len(expert_obj.experts)

        print(f"[{modality.upper()} Modality]: {num_experts} Experts")

        expert_details = ""
        column_details = ""
        # Check expert frozen status
        for idx, expert in enumerate(expert_obj.experts):
            is_frozen = check_frozen_status(expert.parameters())
            status = "Frozen" if is_frozen else "Trainable"
            expert_details += f"Expert {idx}: {status}; "

        # Check router column frozen status
        if hasattr(expert_obj, 'router'):
            for idx, column in enumerate(expert_obj.router.columns):
                is_frozen = not column.requires_grad  # Columns are individual parameters
                status = "Frozen" if is_frozen else "Trainable"
                column_details += f"Column {idx}: {status}; "

        print(expert_details)
        print(column_details)


def print_expert_utils(encoders, names):
    expert_info = 'Expert utilization ==> '
    gate_info = 'Gate utilization ==> '
    for idx, encoder in enumerate(encoders):
        expert_name = names[idx]
        expert_info += expert_name
        expert_info += ' '
        tracker = encoder.tracker
        for layer in range(tracker.num_layers):
            expert_util = tracker.utilization(layer).tolist() # tensor of floats in [0,1]

            expert_info += str([round(u, 2) for u in expert_util])

            gate_util = tracker.gate_utilization(layer).tolist()
            gate_info += str([round(u, 2) for u in gate_util])

            expert_info += '; '
            gate_info += '; '
    print(expert_info)
    print(gate_info)


class XSMoEMMEncoder(nn.Module):
    def __init__(self, args):
        super(XSMoEMMEncoder, self).__init__()
        self.cv_encoder = LadderSMoE(args, 'cv')
        self.text_encoder = LadderSMoE(args, 'text')
        self.com_dense = nn.Linear(2 * args.word_embedding_dim, args.embedding_dim)
        self.args = args

    def forward(self, images, texts):
        # all torch.Size([1408, 768])
        h_states_last_cv, lb_loss_cv = self.cv_encoder(images)
        h_states_last_text, lb_loss_text = self.text_encoder(texts)
        # Concatenate embeddings
        combined = torch.cat([h_states_last_cv, h_states_last_text], dim=1)
        # shape: (batch_size, 2 * embed_dim)
        return self.com_dense(combined), lb_loss_cv + lb_loss_text

    def print_util(self):
        print_expert_utils(self.get_encoders(), ['cv', 'text'])

    def print_structure(self):
        modalities = {
            'cv': self.cv_encoder,
            'text': self.text_encoder,
        }
        print_structure(modalities)

    def get_encoders(self):
        return [self.cv_encoder, self.text_encoder]


