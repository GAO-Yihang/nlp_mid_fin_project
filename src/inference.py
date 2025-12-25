import argparse
import json
import torch
from torch.utils.data import DataLoader

from src.config import PAD_ID, BOS_ID, EOS_ID
from src.data.dataset import JsonlIdsDataset
from src.data.collate import collate_nmt
from src.models.rnn_nmt import Seq2SeqRNN
from src.models.transformer_nmt import TransformerNMT
from src.utils.checkpoint import load_checkpoint
from src.utils.decoding import (
    greedy_decode_rnn, beam_search_decode_rnn,
    greedy_decode_transformer, beam_search_decode_transformer
)
from src.utils.bleu import bleu4_corpus


def load_vocab(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv = {int(i): tok for tok, i in vocab.items()}
    return vocab, inv


def strip_special(ids):
    out = []
    for x in ids:
        if x == EOS_ID:
            break
        if x in (PAD_ID, BOS_ID):
            continue
        out.append(x)
    return out


def ids_to_text(ids, inv_vocab):
    toks = [inv_vocab.get(i, "<unk>") for i in ids]
    return " ".join(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", type=str, required=True, choices=["rnn", "transformer"])
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--test_ids", type=str, required=True, help="path to test/data.ids.jsonl")
    ap.add_argument("--src_vocab", type=str, required=True)
    ap.add_argument("--tgt_vocab", type=str, required=True)

    # scoring
    ap.add_argument("--score", action="store_true", help="compute BLEU-4 using tgt_ids in test jsonl")
    ap.add_argument("--save_metrics", type=str, default="", help="optional path to save metrics.json")

    # RNN params (must match training)
    ap.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"])
    ap.add_argument("--attn_type", type=str, default="additive", choices=["dot", "general", "additive"])
    ap.add_argument("--emb_size", type=int, default=256)
    ap.add_argument("--hidden_size", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Transformer params (must match training)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--num_layers_trf", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout_trf", type=float, default=0.1)
    ap.add_argument("--pos_type", type=str, default="absolute", choices=["absolute", "relative"])
    ap.add_argument("--norm_type", type=str, default="layernorm", choices=["layernorm", "rmsnorm"])

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--decode", type=str, default="beam", choices=["greedy", "beam"])
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=120)

    ap.add_argument("--out", type=str, default="pred.txt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    src_vocab, inv_src = load_vocab(args.src_vocab)
    tgt_vocab, inv_tgt = load_vocab(args.tgt_vocab)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # build model
    if args.model_type == "rnn":
        model = Seq2SeqRNN(
            src_vocab=src_vocab_size, tgt_vocab=tgt_vocab_size,
            emb_size=args.emb_size, hidden_size=args.hidden_size,
            num_layers=args.num_layers, rnn_type=args.rnn_type,
            dropout=args.dropout, attn_type=args.attn_type,
            pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID
        ).to(device)
    else:
        model = TransformerNMT(
            src_vocab=src_vocab_size, tgt_vocab=tgt_vocab_size,
            d_model=args.d_model, n_heads=args.n_heads,
            num_layers=args.num_layers_trf, d_ff=args.d_ff,
            dropout=args.dropout_trf,
            pos_type=args.pos_type, norm_type=args.norm_type,
            pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID
        ).to(device)

    load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()

    ds = JsonlIdsDataset(args.test_ids)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate_nmt(b, PAD_ID), num_workers=0)

    preds_text = []
    refs, hyps = [], []  # for BLEU (list[list[int]])
    total_ref_len = 0
    total_hyp_len = 0

    with torch.no_grad():
        for batch in loader:
            src = batch.src_ids.to(device)
            tgt = batch.tgt_ids.to(device)  # test jsonl has tgt_ids, so we can score

            if args.model_type == "rnn":
                if args.decode == "greedy":
                    pred = greedy_decode_rnn(model, src, max_len=args.max_len)
                else:
                    pred = beam_search_decode_rnn(model, src, beam_size=args.beam_size, max_len=args.max_len)
            else:
                if args.decode == "greedy":
                    pred = greedy_decode_transformer(model, src, max_len=args.max_len)
                else:
                    pred = beam_search_decode_transformer(model, src, beam_size=args.beam_size, max_len=args.max_len)

            for i in range(pred.size(0)):
                hyp_ids = strip_special(pred[i].tolist())
                preds_text.append(ids_to_text(hyp_ids, inv_tgt))

                if args.score:
                    ref_ids = strip_special(tgt[i].tolist())
                    refs.append(ref_ids)
                    hyps.append(hyp_ids)
                    total_ref_len += len(ref_ids)
                    total_hyp_len += len(hyp_ids)

    # save predictions
    with open(args.out, "w", encoding="utf-8") as f:
        for line in preds_text:
            f.write(line + "\n")
    print(f"Saved predictions to: {args.out}")

    # compute BLEU
    if args.score:
        bleu = bleu4_corpus(refs, hyps) * 100.0
        avg_ref_len = total_ref_len / max(1, len(refs))
        avg_hyp_len = total_hyp_len / max(1, len(hyps))
        len_ratio = (total_hyp_len / max(1, total_ref_len))

        metrics = {
            "BLEU4": bleu,
            "decode": args.decode,
            "beam_size": args.beam_size if args.decode == "beam" else 1,
            "avg_ref_len": avg_ref_len,
            "avg_hyp_len": avg_hyp_len,
            "len_ratio(hyp/ref)": len_ratio,
        }

        print(f"[SCORE] BLEU-4 = {bleu:.2f}")
        print(f"[SCORE] avg_ref_len={avg_ref_len:.2f} | avg_hyp_len={avg_hyp_len:.2f} | len_ratio={len_ratio:.3f}")

        if args.save_metrics:
            with open(args.save_metrics, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"Saved metrics to: {args.save_metrics}")


if __name__ == "__main__":
    main()

