from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class NMTBatch:
    src_ids: torch.Tensor  # (B, S)
    tgt_ids: torch.Tensor  # (B, T)

def pad_1d(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out

def collate_nmt(batch: List[Dict[str, Any]], pad_id: int) -> NMTBatch:
    src = [x["src_ids"] for x in batch]
    tgt = [x["tgt_ids"] for x in batch]
    src_t = pad_1d(src, pad_id)
    tgt_t = pad_1d(tgt, pad_id)
    return NMTBatch(src_ids=src_t, tgt_ids=tgt_t)
