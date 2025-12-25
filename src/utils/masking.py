import torch

def make_padding_mask(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    seq: (B, L)
    returns: (B, 1, 1, L) with True=keep, False=mask
    """
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)

def subsequent_mask(size: int, device=None) -> torch.Tensor:
    """
    Causal mask for decoder self-attention
    returns: (1, 1, L, L) True=keep, False=mask
    """
    attn_shape = (1, 1, size, size)
    upper = torch.triu(torch.ones(attn_shape, device=device), diagonal=1).bool()
    return ~upper
