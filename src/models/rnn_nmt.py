from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    dot / general / additive attention
    query: (B,H), keys/values: (B,S,H)
    mask: (B,S) True=keep
    """
    def __init__(self, hidden_size: int, attn_type: str = "additive"):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_type = attn_type.lower()
        assert self.attn_type in ["dot", "general", "additive"]
        if self.attn_type == "general":
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.attn_type == "additive":
            self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Wk = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, H = keys.shape
        if self.attn_type == "dot":
            scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        elif self.attn_type == "general":
            k_proj = self.W(keys)
            scores = torch.bmm(k_proj, query.unsqueeze(2)).squeeze(2)
        else:
            q_proj = self.Wq(query).unsqueeze(1)
            k_proj = self.Wk(keys)
            scores = self.v(torch.tanh(q_proj + k_proj)).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn_w = F.softmax(scores, dim=1)
        context = torch.bmm(attn_w.unsqueeze(1), values).squeeze(1)
        return context, attn_w

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int,
                 num_layers: int = 2, rnn_type: str = "gru", dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        rnn_type = rnn_type.lower()
        assert rnn_type in ["gru", "lstm"]
        self.rnn_type = rnn_type
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

    def forward(self, src_ids: torch.Tensor):
        x = self.dropout(self.emb(src_ids))
        enc_out, enc_h = self.rnn(x)
        return enc_out, enc_h

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int,
                 num_layers: int = 2, rnn_type: str = "gru", dropout: float = 0.1,
                 attn_type: str = "additive", pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        rnn_type = rnn_type.lower()
        assert rnn_type in ["gru", "lstm"]
        self.rnn_type = rnn_type
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.rnn = rnn_cls(
            input_size=emb_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        self.attn = Attention(hidden_size, attn_type=attn_type)
        self.out = nn.Linear(hidden_size * 2, vocab_size)

    def forward_step(self, y_prev: torch.Tensor, dec_state, enc_out: torch.Tensor,
                     src_mask: Optional[torch.Tensor] = None):
        y_emb = self.dropout(self.emb(y_prev))

        if self.rnn_type == "gru":
            query = dec_state[-1]
        else:
            h, c = dec_state
            query = h[-1]

        context, attn_w = self.attn(query=query, keys=enc_out, values=enc_out, mask=src_mask)
        rnn_in = torch.cat([y_emb, context], dim=-1).unsqueeze(1)
        dec_out, new_state = self.rnn(rnn_in, dec_state)
        dec_h = dec_out.squeeze(1)

        logits = self.out(torch.cat([dec_h, context], dim=-1))
        return logits, new_state, attn_w

class Seq2SeqRNN(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, emb_size: int, hidden_size: int,
                 num_layers: int = 2, rnn_type: str = "gru", dropout: float = 0.1,
                 attn_type: str = "additive", pad_id: int = 0, bos_id: int = 2, eos_id: int = 3):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.encoder = RNNEncoder(src_vocab, emb_size, hidden_size, num_layers, rnn_type, dropout, pad_id)
        self.decoder = RNNDecoder(tgt_vocab, emb_size, hidden_size, num_layers, rnn_type, dropout, attn_type, pad_id)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor, teacher_forcing_ratio: float = 1.0):
        B, T = tgt_ids.shape
        enc_out, enc_h = self.encoder(src_ids)
        src_mask = (src_ids != self.pad_id)

        dec_state = enc_h
        y_prev = tgt_ids[:, 0]  # BOS
        logits_all = []

        for t in range(1, T):
            logits, dec_state, _ = self.decoder.forward_step(y_prev, dec_state, enc_out, src_mask)
            logits_all.append(logits.unsqueeze(1))
            use_tf = (torch.rand(1, device=src_ids.device).item() < teacher_forcing_ratio)
            y_prev = tgt_ids[:, t] if use_tf else torch.argmax(logits, dim=-1)

        return torch.cat(logits_all, dim=1)  # (B, T-1, V)
