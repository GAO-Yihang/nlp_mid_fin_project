import argparse
import json
import random
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import PAD_ID, BOS_ID, EOS_ID, DEFAULT_SEED
from src.data.dataset import JsonlIdsDataset
from src.data.collate import collate_nmt
from src.models.rnn_nmt import Seq2SeqRNN
from src.utils.checkpoint import save_checkpoint
from src.utils.bleu import bleu4_corpus
from src.utils.decoding import greedy_decode_rnn, beam_search_decode_rnn


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_special(ids, pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID):
    out = []
    for x in ids:
        if x == eos_id:
            break
        if x in (pad_id, bos_id):
            continue
        out.append(x)
    return out


def maybe_load_inv_vocab(vocab_json_path: str):
    """
    Optional: for TensorBoard text samples.
    vocab json: {"<pad>":0, ...}
    returns inv: {0:"<pad>", ...}
    """
    if not vocab_json_path:
        return None
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv = {int(i): tok for tok, i in vocab.items()}
    return inv


def ids_to_text(ids, inv_vocab):
    if inv_vocab is None:
        return " ".join(str(i) for i in ids)  # fallback: ids
    return " ".join(inv_vocab.get(i, "<unk>") for i in ids)


@torch.no_grad()
def evaluate_bleu_rnn(
    model, loader, device, decode="beam", beam_size=5, max_len=120,
    log_samples: int = 0,
    inv_src_vocab=None, inv_tgt_vocab=None
):
    """
    Returns:
      bleu (float in [0,1]),
      samples (list[str]) optional for tensorboard
    """
    model.eval()
    refs, hyps = [], []
    sample_texts = []

    for batch_idx, batch in enumerate(loader):
        src = batch.src_ids.to(device)
        tgt = batch.tgt_ids.to(device)

        if decode == "greedy":
            pred = greedy_decode_rnn(model, src, max_len=max_len)
        else:
            pred = beam_search_decode_rnn(model, src, beam_size=beam_size, max_len=max_len)

        for i in range(src.size(0)):
            ref = strip_special(tgt[i].tolist())
            hyp = strip_special(pred[i].tolist())
            refs.append(ref)
            hyps.append(hyp)

            if log_samples > 0 and len(sample_texts) < log_samples:
                src_txt = ids_to_text(strip_special(src[i].tolist(), pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID), inv_src_vocab)
                ref_txt = ids_to_text(ref, inv_tgt_vocab)
                hyp_txt = ids_to_text(hyp, inv_tgt_vocab)
                sample_texts.append(
                    f"[sample {len(sample_texts)}]\nSRC: {src_txt}\nREF: {ref_txt}\nHYP: {hyp_txt}\n"
                )

    bleu = bleu4_corpus(refs, hyps)
    return bleu, sample_texts


def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--train_ids", type=str, required=True, help="path to train/data.ids.jsonl")
    ap.add_argument("--valid_ids", type=str, required=True, help="path to valid/data.ids.jsonl")
    ap.add_argument("--src_vocab_size", type=int, required=True)
    ap.add_argument("--tgt_vocab_size", type=int, required=True)

    # optional vocab for pretty TensorBoard samples
    ap.add_argument("--src_vocab_json", type=str, default="/data/250010229/phd_hw/AP0004/midfin/data/train/vocab.src.json", help="optional vocab.src.json for TB text")
    ap.add_argument("--tgt_vocab_json", type=str, default="/data/250010229/phd_hw/AP0004/midfin/data/train/vocab.tgt.json", help="optional vocab.tgt.json for TB text")

    # model
    ap.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"])
    ap.add_argument("--attn_type", type=str, default="additive", choices=["dot", "general", "additive"])
    ap.add_argument("--emb_size", type=int, default=256)
    ap.add_argument("--hidden_size", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # train
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--teacher_forcing", type=float, default=1.0, help="teacher forcing ratio (0~1)")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # decoding for valid BLEU
    ap.add_argument("--decode", type=str, default="beam", choices=["greedy", "beam"])
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=120)

    # checkpoint
    ap.add_argument("--save_path", type=str, default="checkpoints/rnn_best.pt")

    # tensorboard
    ap.add_argument("--logdir", type=str, default="/data/250010229/phd_hw/AP0004/midfin/src/tensorboard_logs/rnn")
    ap.add_argument("--exp_name", type=str, default="", help="subfolder name under logdir; default auto")
    ap.add_argument("--log_every", type=int, default=200, help="log train scalars every N steps")
    ap.add_argument("--log_samples", type=int, default=5, help="how many valid samples to log as text each epoch")

    # misc
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TensorBoard writer
    if args.exp_name:
        run_name = args.exp_name
    else:
        # auto run name
        run_name = f"rnn_{args.rnn_type}_attn-{args.attn_type}_hs{args.hidden_size}_emb{args.emb_size}_tf{args.teacher_forcing}"
    writer = SummaryWriter(log_dir=f"{args.logdir}/{run_name}")

    # record hparams early
    writer.add_text("meta/run_name", run_name)
    writer.add_text("meta/device", device)
    writer.add_text("meta/args", "\n".join([f"{k}: {v}" for k, v in vars(args).items()]))

    inv_src = maybe_load_inv_vocab(args.src_vocab_json)
    inv_tgt = maybe_load_inv_vocab(args.tgt_vocab_json)

    # data
    train_ds = JsonlIdsDataset(args.train_ids)
    valid_ds = JsonlIdsDataset(args.valid_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_nmt(b, PAD_ID), num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_nmt(b, PAD_ID), num_workers=0)

    # model
    model = Seq2SeqRNN(
        src_vocab=args.src_vocab_size,
        tgt_vocab=args.tgt_vocab_size,
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        attn_type=args.attn_type,
        pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_bleu = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        start_t = time.time()

        for step, batch in enumerate(train_loader, start=1):
            src = batch.src_ids.to(device)
            tgt = batch.tgt_ids.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(src, tgt, teacher_forcing_ratio=args.teacher_forcing)  # (B,T-1,V)
            V = logits.size(-1)

            gold = tgt[:, 1:]  # (B,T-1)
            loss = F.cross_entropy(logits.reshape(-1, V), gold.reshape(-1), ignore_index=PAD_ID)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            optim.step()

            with torch.no_grad():
                nonpad = (gold != PAD_ID).sum().item()

            # token-weighted loss sum for perplexity
            epoch_loss_sum += loss.item() * nonpad
            epoch_tokens += nonpad

            global_step += 1

            # TB: step-level logs
            if global_step % args.log_every == 0:
                avg_loss = epoch_loss_sum / max(1, epoch_tokens)
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/teacher_forcing", args.teacher_forcing, global_step)
                writer.add_scalar("train/tokens_per_step", nonpad, global_step)

        # epoch summary
        avg_loss_epoch = epoch_loss_sum / max(1, epoch_tokens)
        train_ppl = torch.exp(torch.tensor(avg_loss_epoch)).item()

        # valid bleu + samples
        bleu, samples = evaluate_bleu_rnn(
            model, valid_loader, device,
            decode=args.decode, beam_size=args.beam_size, max_len=args.max_len,
            log_samples=args.log_samples,
            inv_src_vocab=inv_src, inv_tgt_vocab=inv_tgt
        )

        elapsed = time.time() - start_t

        print(f"[RNN] Epoch {epoch:02d} | loss={avg_loss_epoch:.4f} | train_ppl={train_ppl:.3f} | "
              f"valid_BLEU4={bleu*100:.2f} | time={elapsed:.1f}s")

        # TB: epoch-level logs
        writer.add_scalar("epoch/train_loss", avg_loss_epoch, epoch)
        writer.add_scalar("epoch/train_ppl", train_ppl, epoch)
        writer.add_scalar("epoch/valid_bleu4", bleu * 100.0, epoch)
        writer.add_scalar("epoch/epoch_time_sec", elapsed, epoch)

        # TB: text samples
        if samples:
            writer.add_text("valid/samples", "\n\n".join(samples), epoch)

        # save best
        if bleu > best_bleu:
            best_bleu = bleu
            save_checkpoint(
                args.save_path, model, optimizer=optim,
                meta={"epoch": epoch, "best_bleu": best_bleu, "args": vars(args)}
            )
            print(f"  -> saved best to {args.save_path} (BLEU4={best_bleu*100:.2f})")

    # record final hparams (TensorBoard HParams tab needs metrics dict)
    try:
        writer.add_hparams(
            hparam_dict={
                "rnn_type": args.rnn_type,
                "attn_type": args.attn_type,
                "emb_size": args.emb_size,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "teacher_forcing": args.teacher_forcing,
                "beam_size": args.beam_size,
            },
            metric_dict={"best_bleu4": best_bleu * 100.0},
        )
    except Exception:
        # some TB versions can be picky; scalars above are enough
        pass

    writer.close()


if __name__ == "__main__":
    main()

