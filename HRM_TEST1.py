import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class HRMConfig:
    vocab_size: int = 10
    embed_dim: int = 128
    num_heads: int = 8
    ff_hidden_dim: int = 512
    num_high_layers: int = 2
    num_low_layers: int = 4
    seq_len: int = 81 
    dropout: float = 0.1
    max_pos: int = 81
    use_batch_first: bool = True
    q_head_hidden: int = 64
    act_max_steps: int = 20  
    act_epsilon: float = 0.01
    deep_supervision: bool = True

class HierarchicalReasoningModelBlock(nn.Module):

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config

        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=config.use_batch_first
        )

        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):

        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=attn_mask)
        x = x + attn_out

        x_ln = self.ln2(x)
        ff_out = self.ff(x_ln)
        x = x + ff_out

        return x


class HierarchicalReasoningModelLayer(nn.Module):

    def __init__(self, config: HRMConfig, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([HierarchicalReasoningModelBlock(config) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x


class HierarchicalReasoningModelCore(nn.Module):

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_pos, config.embed_dim)

        self.high_stack = HierarchicalReasoningModelLayer(config, num_layers=config.num_high_layers)
        self.low_stack = HierarchicalReasoningModelLayer(config, num_layers=config.num_low_layers)

        self.final_ln = nn.LayerNorm(config.embed_dim)

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

        self.q_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.q_head_hidden),
            nn.ReLU(),
            nn.Linear(config.q_head_hidden, 1)
        )

        self.dropout = nn.Dropout(config.dropout)
        
        self.deep_supervision_heads = nn.ModuleList([
            nn.Linear(config.embed_dim, config.vocab_size)
            for _ in range(config.num_high_layers + config.num_low_layers)
        ]) if config.deep_supervision else None
        
        self.halting_unit = nn.Linear(config.embed_dim, 1)

    def act_forward(self, x, max_steps=20, threshold=0.01):
        halting_probs = []
        remainders = []
        n_updates = []

        batch, seq = x.shape[0], x.shape[1]
        halt = torch.zeros(batch, seq, 1, device=x.device)
        step = 0

        while (halt < 1.0 - threshold).any() and step < max_steps:
            p = torch.sigmoid(self.halting_unit(x))
            halting_probs.append(p)

            halt = halt + p * (1 - halt)
            remainders.append((1 - halt).clone())
            n_updates.append(step + 1)

            step += 1

        return x, halting_probs, remainders, n_updates

    def forward(self, input_tokens: torch.LongTensor, return_intermediate: bool = False):
        batch, seq = input_tokens.shape
        assert seq == self.config.seq_len, f"expected seq_len={self.config.seq_len}, got {seq}"

        tok_emb = self.token_embed(input_tokens)
        positions = torch.arange(seq, device=input_tokens.device).unsqueeze(0).expand(batch, -1)
        pos_emb = self.pos_embed(positions)
        x = tok_emb + pos_emb
        x = self.dropout(x)

        intermediate_outputs = []

        identity = x
        x_high, halt_probs, remainders, n_updates = self.act_forward(
            x, self.config.act_max_steps, self.config.act_epsilon
        )
        x = self.final_ln(identity + x_high)

        for i, layer in enumerate(self.high_stack.layers):
            identity = x
            x = layer(x)
            x = x + identity
            if self.config.deep_supervision:
                intermediate_outputs.append(self.deep_supervision_heads[i](x))
        for i, layer in enumerate(self.low_stack.layers):
            x = layer(x)
            if self.config.deep_supervision:
                intermediate_outputs.append(
                    self.deep_supervision_heads[i + self.config.num_high_layers](x)
                )

        x = self.final_ln(x)
        lm_logits = self.lm_head(x)

        pooled = x.mean(dim=1)
        q_values = self.q_head(pooled)

        backtracking_count = len(n_updates) if n_updates else 0
        if halt_probs:
            participation_ratio = sum([(1.0 - h).mean().item() for h in halt_probs]) / len(halt_probs)
        else:
            participation_ratio = 0.0

        if return_intermediate:
            return lm_logits, q_values, {
                "hidden": x,
                "pooled": pooled,
                "intermediate_outputs": intermediate_outputs,
                "halting_probs": halt_probs,
                "remainders": remainders,
                "n_updates": n_updates,
                "backtracking_count": backtracking_count,
                "participation_ratio": participation_ratio
            }
        return lm_logits, q_values
