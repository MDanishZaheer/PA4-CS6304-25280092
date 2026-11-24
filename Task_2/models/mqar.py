# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""A fixed two-block pre-normalized decoder for query-position classification."""

import math

import torch
from torch import nn

from Task_2.attention.operators import attention_operator


class RecallBlock(nn.Module):
    """Share the same normalization, projections and MLP across attention mechanisms."""

    def __init__(self, settings, method, content_k=64):
        super().__init__()
        hidden = settings["hidden_dimension"]
        self.heads = settings["heads"]
        self.dimension = hidden // self.heads
        self.method, self.content_k, self.settings = method, content_k, settings
        self.attention_norm = nn.LayerNorm(hidden)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = [nn.Linear(hidden, hidden) for _ in range(4)]
        self.mlp_norm = nn.LayerNorm(hidden)
        width = hidden * settings["mlp_multiplier"]
        self.mlp = nn.Sequential(nn.Linear(hidden, width), nn.GELU(), nn.Linear(width, hidden))

    def forward(self, hidden, override_method=None):
        normalized = self.attention_norm(hidden)
        batch, length, width = hidden.shape
        q, k, v = [projection(normalized).reshape(batch, length, self.heads, self.dimension).transpose(1, 2)
                   for projection in [self.q_proj, self.k_proj, self.v_proj]]
        attention = attention_operator(q, k, v, override_method or self.method,
            window=self.settings["window"], content_k=self.content_k,
            chunk_size=self.settings["linear_chunk"], epsilon=self.settings["linear_epsilon"])
        hidden = hidden + self.o_proj(attention.transpose(1, 2).reshape(batch, length, width))
        return hidden + self.mlp(self.mlp_norm(hidden))


class RecallTransformer(nn.Module):
    """Use 256 input IDs and the 64 valid value IDs as the classification label space."""

    def __init__(self, settings, method, content_k=64):
        super().__init__()
        self.settings, self.method, self.content_k = settings, method, content_k
        hidden, length = settings["hidden_dimension"], settings["sequence_length"]
        self.embedding = nn.Embedding(settings["vocabulary"], hidden)
        positions = torch.arange(length).float()[:, None]
        frequencies = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.) / hidden))
        encoding = torch.zeros(length, hidden)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(positions * frequencies), torch.cos(positions * frequencies)
        self.register_buffer("positions", encoding[None], persistent=True)
        self.blocks = nn.ModuleList([RecallBlock(settings, method, content_k) for _ in range(settings["layers"])])
        self.final_norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, settings["value_classes"])

    def forward(self, token_ids, query_positions, override_method=None):
        hidden = self.embedding(token_ids) + self.positions[:, :token_ids.shape[1]].to(self.embedding.weight.dtype)
        for block in self.blocks:
            hidden = block(hidden, override_method)
        selected = self.final_norm(hidden).gather(1, query_positions[..., None].expand(-1, -1, hidden.shape[-1]))
        return self.classifier(selected)

    def parameter_count(self):
        """Count actual trainable parameters for the intervention's 5% check."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
