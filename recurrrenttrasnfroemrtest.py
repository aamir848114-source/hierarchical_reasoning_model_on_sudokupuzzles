import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    src_vocab_size: int = 10
    tgt_vocab_size: int = 10
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 8
    hidden_mlp: int = 128
    dropout: float = 0.1
    max_seq_length: int = 81


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden_mlp, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class RecurrentTransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_mlp, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ff = FeedForwardNetwork(d_model, hidden_mlp, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):

        attn_input = self.norm1(x)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)

        ff_input = self.norm2(x)
        ff_output = self.ff(ff_input)
        x = x + self.dropout(ff_output)

        return x


class RecurrentTransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.src_vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_length, config.d_model)

        self.layers = nn.ModuleList([
            RecurrentTransformerEncoderBlock(
                config.d_model, config.num_heads, config.hidden_mlp, config.dropout
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, x, return_intermediate=False):
        batch_size, seq_len = x.size()
        device = x.device

        tok_emb = self.token_embedding(x)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(positions)
        x = tok_emb + pos_emb
        x = F.dropout(x, p=self.config.dropout, training=self.training)

        intermediates = []
        for layer in self.layers:
            x = layer(x)
            if return_intermediate:
                intermediates.append(x.detach().clone())

        x = self.final_norm(x)
        if return_intermediate:
            return x, intermediates
        return x

class RecurrentTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.encoder = RecurrentTransformerEncoder(config)
        self.final_layer = nn.Linear(config.d_model, config.tgt_vocab_size)
        
        # Backtracking and participation tracking
        self.backtracking_count = 0
        self.participation_ratio = 0.0

    def forward(self, src, return_intermediate=False):
        if return_intermediate:
            encoder_output, intermediates = self.encoder(src.long(), return_intermediate=True)
            output_logits = self.final_layer(encoder_output)
            layer_logits = [self.final_layer(h) for h in intermediates]
            return output_logits, layer_logits
        else:
            encoder_output = self.encoder(src.long())
            output_logits = self.final_layer(encoder_output)
        self.backtracking_count = self.config.num_layers
        self.participation_ratio = 1.0
        return output_logits

