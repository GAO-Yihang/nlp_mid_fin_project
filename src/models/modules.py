import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

def NormLayer(norm_type: str, d_model: int):
    norm_type = norm_type.lower()
    assert norm_type in ["layernorm", "rmsnorm"]
    return nn.LayerNorm(d_model) if norm_type == "layernorm" else RMSNorm(d_model)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets: int, num_heads: int, max_distance: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        rp = relative_position
        n = -rp
        n = torch.clamp(n, min=0)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            (torch.log(n.float() / max_exact + 1e-9) / math.log(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n.long(), val_if_large)

    def forward(self, qlen: int, klen: int, device=None) -> torch.Tensor:
        ctx = torch.arange(qlen, device=device)[:, None]
        mem = torch.arange(klen, device=device)[None, :]
        rel = mem - ctx
        buckets = self._relative_position_bucket(rel)
        bias = self.relative_attention_bias(buckets)  # (qlen,klen,heads)
        return bias.permute(2, 0, 1).unsqueeze(0)      # (1, heads, qlen, klen)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_rel_pos_bias: bool = False, rel_pos_buckets: int = 32):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        self.use_rel_pos_bias = use_rel_pos_bias
        self.rel_bias = RelativePositionBias(rel_pos_buckets, n_heads) if use_rel_pos_bias else None

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        B, Lq, _ = q.shape
        B, Lk, _ = k.shape

        q = self.Wq(q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(k).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(v).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if self.use_rel_pos_bias:
            scores = scores + self.rel_bias(Lq, Lk, device=scores.device)

        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask, -1e9)

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.Wo(out)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
