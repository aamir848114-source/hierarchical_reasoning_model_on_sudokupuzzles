from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class RRNConfig:
    input_dim: int
    hidden_dim: int
    message_dim: int
    edge_dim: int
    output_dim: int
    num_steps: int = 3

class RRNNode(nn.Module):
    def __init__(self, hidden_dim: int, msg_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim

        self.mlp_updation = nn.Sequential(
            nn.Linear(hidden_dim + msg_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, h_i: torch.Tensor, m_i: torch.Tensor):

        input_vec = torch.cat([h_i, m_i], dim=-1)
        h_i_new = self.mlp_updation(input_vec)
        return h_i_new

class RRNLayer(nn.Module):
    def __init__(self, hidden_dim: int, message_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.edge_dim = edge_dim

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, message_dim),
            nn.ReLU()
        )

        self.node_update = RRNNode(hidden_dim, message_dim)

    def forward(self, H: torch.Tensor, E: torch.Tensor):

        B, N, D = H.shape

        Hi = H.unsqueeze(2).repeat(1, 1, N, 1)
        Hj = H.unsqueeze(1).repeat(1, N, 1, 1)

        pair_input = torch.cat([Hi, Hj, E], dim=-1)

        M_ij = self.message_mlp(pair_input)

        M_i = M_ij.sum(dim=2)

        H_new = self.node_update(H, M_i)
        return H_new

class RecurrentRelationalNetwork(nn.Module):
    def __init__(self, config: RRNConfig):
        super().__init__()
        self.config = config

        self.encoder = nn.Linear(config.input_dim, config.hidden_dim)
        self.rrn_layer = RRNLayer(config.hidden_dim, config.message_dim, config.edge_dim)
        self.decoder = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, X: torch.Tensor, E: torch.Tensor, return_intermediate=False):
        H = F.relu(self.encoder(X))
        intermediates = []
        for _ in range(self.config.num_steps):
            H = self.rrn_layer(H, E)
            if return_intermediate:
                intermediates.append(H.detach().clone())
        out = self.decoder(H)
        self.backtracking_count = self.config.num_steps
        self.participation_ratio = 1.0
        if return_intermediate:
            # Keep intermediate logits in the model's default dtype (float32)
            layer_logits = [self.decoder(h_i) for h_i in intermediates]
            return out, layer_logits
        return out
