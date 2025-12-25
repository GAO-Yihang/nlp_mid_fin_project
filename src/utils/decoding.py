from typing import Optional
import torch
import torch.nn.functional as F
#from masking import make_padding_mask, subsequent_mask

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

# -------------------------
# RNN decoding
# -------------------------
@torch.no_grad()
def greedy_decode_rnn(model, src_ids: torch.Tensor, max_len: int = 100) -> torch.Tensor:
    model.eval()
    B = src_ids.size(0)
    enc_out, enc_h = model.encoder(src_ids)
    src_mask = (src_ids != model.pad_id)

    dec_state = enc_h
    y_prev = torch.full((B,), model.bos_id, device=src_ids.device, dtype=torch.long)

    outs = [y_prev.unsqueeze(1)]
    finished = torch.zeros(B, device=src_ids.device).bool()

    for _ in range(max_len):
        logits, dec_state, _ = model.decoder.forward_step(y_prev, dec_state, enc_out, src_mask)
        y_prev = torch.argmax(logits, dim=-1)
        outs.append(y_prev.unsqueeze(1))
        finished |= (y_prev == model.eos_id)
        if finished.all():
            break

    return torch.cat(outs, dim=1)

@torch.no_grad()
def beam_search_decode_rnn(model, src_ids: torch.Tensor, beam_size: int = 5,
                          max_len: int = 100, len_penalty: float = 0.6) -> torch.Tensor:
    model.eval()
    device = src_ids.device
    B, S = src_ids.shape
    enc_out, enc_h = model.encoder(src_ids)
    src_mask = (src_ids != model.pad_id)

    # expand for beam
    enc_out = enc_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(B * beam_size, S, -1)
    src_mask = src_mask.unsqueeze(1).repeat(1, beam_size, 1).view(B * beam_size, S)

    def expand_state(state):
        if model.decoder.rnn_type == "gru":
            # (L,B,H) -> (L,B*K,H)
            return state.unsqueeze(2).repeat(1, 1, beam_size, 1).view(state.size(0), B * beam_size, state.size(-1))
        else:
            h, c = state
            h2 = h.unsqueeze(2).repeat(1, 1, beam_size, 1).view(h.size(0), B * beam_size, h.size(-1))
            c2 = c.unsqueeze(2).repeat(1, 1, beam_size, 1).view(c.size(0), B * beam_size, c.size(-1))
            return (h2, c2)

    dec_state = expand_state(enc_h)

    beams = torch.full((B, beam_size, 1), model.bos_id, device=device, dtype=torch.long)
    beam_scores = torch.zeros(B, beam_size, device=device)
    beam_scores[:, 1:] = -1e9
    finished = torch.zeros(B, beam_size, device=device).bool()

    y_prev = beams[:, :, -1].reshape(B * beam_size)

    for _ in range(max_len):
        logits, dec_state, _ = model.decoder.forward_step(y_prev, dec_state, enc_out, src_mask)
        logp = F.log_softmax(logits, dim=-1).view(B, beam_size, -1)

        logp = logp.masked_fill(finished.unsqueeze(-1), -1e9)

        total = beam_scores.unsqueeze(-1) + logp
        V = total.size(-1)
        total = total.view(B, beam_size * V)

        top_scores, top_idx = torch.topk(total, k=beam_size, dim=-1)
        old_beam = top_idx // V
        next_tok = top_idx % V

        gathered = torch.gather(beams, 1, old_beam.unsqueeze(-1).repeat(1, 1, beams.size(-1)))
        beams = torch.cat([gathered, next_tok.unsqueeze(-1)], dim=-1)

        beam_scores = top_scores
        finished = torch.gather(finished, 1, old_beam) | (next_tok == model.eos_id)

        # reorder states
        if model.decoder.rnn_type == "gru":
            h = dec_state
            h = h.view(h.size(0), B, beam_size, -1)
            idx = old_beam.unsqueeze(0).unsqueeze(-1).repeat(h.size(0), 1, 1, h.size(-1))
            h = torch.gather(h, 2, idx)
            dec_state = h.view(h.size(0), B * beam_size, -1)
        else:
            h, c = dec_state
            h = h.view(h.size(0), B, beam_size, -1)
            c = c.view(c.size(0), B, beam_size, -1)
            idx = old_beam.unsqueeze(0).unsqueeze(-1).repeat(h.size(0), 1, 1, h.size(-1))
            h = torch.gather(h, 2, idx)
            c = torch.gather(c, 2, idx)
            dec_state = (h.view(h.size(0), B * beam_size, -1),
                         c.view(c.size(0), B * beam_size, -1))

        y_prev = beams[:, :, -1].reshape(B * beam_size)

        if finished.all():
            break

    lengths = beams.size(-1) * torch.ones((B, beam_size), device=device)
    lp = ((5.0 + lengths) / 6.0) ** len_penalty
    final_scores = beam_scores / lp
    best = torch.argmax(final_scores, dim=1)
    return beams[torch.arange(B, device=device), best]


# -------------------------
# Transformer decoding
# -------------------------
@torch.no_grad()
def greedy_decode_transformer(model, src_ids: torch.Tensor, max_len: int = 100) -> torch.Tensor:
    model.eval()
    B = src_ids.size(0)
    memory, src_kpm = model.encode(src_ids)

    ys = torch.full((B, 1), model.bos_id, device=src_ids.device, dtype=torch.long)
    for _ in range(max_len):
        tgt_kpm = make_padding_mask(ys, model.pad_id)
        tgt_causal = subsequent_mask(ys.size(1), device=ys.device)
        logits = model.decode(ys, memory, tgt_kpm=tgt_kpm, tgt_causal=tgt_causal, src_kpm=src_kpm)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        if (next_tok.squeeze(1) == model.eos_id).all():
            break
    return ys

@torch.no_grad()
def beam_search_decode_transformer(model, src_ids: torch.Tensor, beam_size: int = 5,
                                  max_len: int = 100, len_penalty: float = 0.6) -> torch.Tensor:
    model.eval()
    device = src_ids.device
    B = src_ids.size(0)
    memory, src_kpm = model.encode(src_ids)

    memory = memory.unsqueeze(1).repeat(1, beam_size, 1, 1).view(B * beam_size, memory.size(1), memory.size(2))
    src_kpm = src_kpm.unsqueeze(1).repeat(1, beam_size, 1, 1, 1).view(B * beam_size, 1, 1, src_kpm.size(-1))

    ys = torch.full((B, beam_size, 1), model.bos_id, device=device, dtype=torch.long)
    beam_scores = torch.zeros(B, beam_size, device=device)
    beam_scores[:, 1:] = -1e9
    finished = torch.zeros(B, beam_size, device=device).bool()

    for _ in range(max_len):
        flat = ys.view(B * beam_size, -1)
        tgt_kpm = make_padding_mask(flat, model.pad_id)
        tgt_causal = subsequent_mask(flat.size(1), device=device)
        logits = model.decode(flat, memory, tgt_kpm=tgt_kpm, tgt_causal=tgt_causal, src_kpm=src_kpm)
        logp = F.log_softmax(logits[:, -1, :], dim=-1).view(B, beam_size, -1)

        logp = logp.masked_fill(finished.unsqueeze(-1), -1e9)

        total = beam_scores.unsqueeze(-1) + logp
        V = total.size(-1)
        total = total.view(B, beam_size * V)
        top_scores, top_idx = torch.topk(total, k=beam_size, dim=-1)

        old_beam = top_idx // V
        next_tok = top_idx % V

        gathered = torch.gather(ys, 1, old_beam.unsqueeze(-1).repeat(1, 1, ys.size(-1)))
        ys = torch.cat([gathered, next_tok.unsqueeze(-1)], dim=-1)

        beam_scores = top_scores
        finished = torch.gather(finished, 1, old_beam) | (next_tok == model.eos_id)

        if finished.all():
            break

    lengths = ys.size(-1) * torch.ones((B, beam_size), device=device)
    lp = ((5.0 + lengths) / 6.0) ** len_penalty
    final_scores = beam_scores / lp
    best = torch.argmax(final_scores, dim=1)
    return ys[torch.arange(B, device=device), best]
