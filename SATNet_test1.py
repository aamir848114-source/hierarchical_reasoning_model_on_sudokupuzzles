import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class SATNetConfig:
    n_vars: int = 729
    n_constraints: int = 324
    hidden_dim: int = 512
    max_iter: int = 50
    lam: float = 1e-4
    dropout: float = 0.1

def build_sudoku_constraint_matrix():

    A = torch.zeros((324, 729))
    constraint_idx = 0

    for cell in range(81):
        for d in range(9):
            A[constraint_idx, cell * 9 + d] = 1
        constraint_idx += 1

    for row in range(9):
        for d in range(9):
            for col in range(9):
                A[constraint_idx, (row * 9 + col) * 9 + d] = 1
            constraint_idx += 1

    for col in range(9):
        for d in range(9):
            for row in range(9):
                A[constraint_idx, (row * 9 + col) * 9 + d] = 1
            constraint_idx += 1

    for box_row in range(3):
        for box_col in range(3):
            for d in range(9):
                for i in range(3):
                    for j in range(3):
                        row = box_row * 3 + i
                        col = box_col * 3 + j
                        A[constraint_idx, (row * 9 + col) * 9 + d] = 1
                constraint_idx += 1

    return A

class SATNet(nn.Module):
    def __init__(self, config: SATNetConfig):
        super().__init__()
        self.config = config
        self.register_buffer('A', build_sudoku_constraint_matrix())
        self.lam = config.lam

    def forward(self, z_init):

        A = self.A
        lam = self.lam
        batch_size = z_init.size(0)
        z = z_init

        for _ in range(self.config.max_iter):
            c = torch.matmul(z, A.t()) - 1
            grad = torch.matmul(c, A)
            z = z - lam * grad
            z = torch.clamp(z, 0, 1)

        return z

class SudokuSATNet(nn.Module):
    def __init__(self, config: SATNetConfig):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(729, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 729)
        )

        self.satnet = SATNet(config)

        self.decoder = nn.Sequential(
            nn.Linear(729, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 729)
        )
        
        # Backtracking and participation tracking
        self.backtracking_count = 0
        self.participation_ratio = 0.0

    def forward(self, x, return_intermediate=False):
        batch_size = x.size(0)
        h = self.encoder(x)
        intermediates = []
        z = h
        for i in range(self.config.max_iter):
            A = self.satnet.A
            lam = self.satnet.lam
            c = torch.matmul(z, A.t()) - 1
            grad = torch.matmul(c, A)
            z = z - lam * grad
            z = torch.clamp(z, 0, 1)
            if return_intermediate and (i % 10 == 0 or i == self.config.max_iter - 1):
                intermediates.append(z.detach().clone())
        logits = self.decoder(z)
        self.backtracking_count = self.config.max_iter
        self.participation_ratio = 1.0
        if return_intermediate:
            layer_logits = [self.decoder(z_i).view(batch_size, 81, 9) for z_i in intermediates]
            return logits.view(batch_size, 81, 9), layer_logits
        return logits.view(batch_size, 81, 9)