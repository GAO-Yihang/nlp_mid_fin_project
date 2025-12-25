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
from src.models.transformer_nmt import TransformerNMT
from src.utils.checkpoint import save_checkpoint
from src.utils.bleu import bleu4_corpus
from src.utils.decoding import greedy_decode_transformer, beam_search_decode_transformer


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
    if not vocab_json_path:
        return None
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {int(i): tok for tok, i in vocab.items()}


def ids_to_text(ids, inv_vocab):
    if inv_vocab is None:
        return " ".join(str(i) for i in ids)
    return " ".join(inv_vocab.get(i, "<unk>") for i in ids)


def label_smoothed_nll_loss(logits, target, ignore_index: int, eps: float):
    """
    logits: (B*(T-1), V)
    target: (B*(T-1),)
    """
    logp = F.log_softmax(logits, dim=-1)  # (N,V)

    nll = F.nll_loss(logp, target, reduction="none", ignore_index=ignore_index)  # (N,)
    smooth = -logp.mean(dim=-1)  # (N,)

    # ignore padding in both terms
    if ignore_index is not None:
        pad_mask = (target == ignore_index)
        nll = nll.masked_fill(pad_mask, 0.0)
        smooth = smooth.masked_fill(pad_mask, 0.0)
        denom = (~pad_mask).sum().clamp_min(1)
    else:
        denom = target.numel()

    loss = ((1.0 - eps) * nll + eps * smooth).sum() / denom
    return loss


@torch.no_grad()
def evaluate_bleu_transformer(
    model, loader, device, decode="beam", beam_size=5, max_len=120,
    log_samples: int = 0, inv_src_vocab=None, inv_tgt_vocab=None
):
    model.eval()
    refs, hyps = [], []
    sample_texts = []

    for batch in loader:
        src = batch.src_ids.to(device)
        tgt = batch.tgt_ids.to(device)

        if decode == "greedy":
            pred = greedy_decode_transformer(model, src, max_len=max_len)
        else:
            pred = beam_search_decode_transformer(model, src, beam_size=beam_size, max_len=max_len)

        for i in range(src.size(0)):
            ref = strip_special(tgt[i].tolist())
            hyp = strip_special(pred[i].tolist())
            refs.append(ref)
            hyps.append(hyp)

            if log_samples > 0 and len(sample_texts) < log_samples:
                src_txt = ids_to_text(strip_special(src[i].tolist()), inv_src_vocab)
                ref_txt = ids_to_text(ref, inv_tgt_vocab)
                hyp_txt = ids_to_text(hyp, inv_tgt_vocab)
                sample_texts.append(
                    f"[sample {len(sample_texts)}]\nSRC: {src_txt}\nREF: {ref_txt}\nHYP: {hyp_txt}\n"
                )

    bleu = bleu4_corpus(refs, hyps)  # returns [0,1]
    return bleu, sample_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ids", type=str, required=True)
    ap.add_argument("--valid_ids", type=str, required=True)
    ap.add_argument("--src_vocab_size", type=int, required=True)
    ap.add_argument("--tgt_vocab_size", type=int, required=True)

    # optional vocab for TB text samples
    ap.add_argument("--src_vocab_json", type=str, default="/data/250010229/phd_hw/AP0004/midfin/data/train/vocab.src.json")
    ap.add_argument("--tgt_vocab_json", type=str, default="/data/250010229/phd_hw/AP0004/midfin/data/train/vocab.tgt.json")

    # model
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Ablations
    ap.add_argument("--pos_type", type=str, default="absolute", choices=["absolute", "relative"])
    ap.add_argument("--norm_type", type=str, default="layernorm", choices=["layernorm", "rmsnorm"])

    # train
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_steps", type=int, default=4000)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # eval decode
    ap.add_argument("--decode", type=str, default="beam", choices=["greedy", "beam"])
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=120)

    # ckpt
    ap.add_argument("--save_path", type=str, default="checkpoints/transformer_best.pt")

    # tensorboard
    ap.add_argument("--logdir", type=str, default="/data/250010229/phd_hw/AP0004/midfin/src/tensorboard_logs/transformer")
    ap.add_argument("--exp_name", type=str, default="")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--log_samples", type=int, default=5)

    # misc
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TensorBoard
    if args.exp_name:
        run_name = args.exp_name
    else:
        run_name = f"trf_L{args.num_layers}_dm{args.d_model}_h{args.n_heads}_pos{args.pos_type}_norm{args.norm_type}_ls{args.label_smoothing}"
    writer = SummaryWriter(log_dir=f"{args.logdir}/{run_name}")
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
    model = TransformerNMT(
        src_vocab=args.src_vocab_size,
        tgt_vocab=args.tgt_vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pos_type=args.pos_type,
        norm_type=args.norm_type,
        pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # warmup: linear warmup to base lr
    def get_lr(step: int):
        if args.warmup_steps <= 0:
            return args.lr
        return args.lr * min(1.0, step / args.warmup_steps)

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

            logits = model(src, tgt)  # (B, T-1, V)
            V = logits.size(-1)
            gold = tgt[:, 1:]         # (B, T-1)

            # flatten
            logits_flat = logits.reshape(-1, V)
            gold_flat = gold.reshape(-1)

            if args.label_smoothing > 0.0:
                loss = label_smoothed_nll_loss(
                    logits_flat, gold_flat, ignore_index=PAD_ID, eps=args.label_smoothing
                )
            else:
                loss = F.cross_entropy(logits_flat, gold_flat, ignore_index=PAD_ID)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()

            # lr warmup
            global_step += 1
            lr_now = get_lr(global_step)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            optim.step()

            with torch.no_grad():
                nonpad = (gold != PAD_ID).sum().item()

            epoch_loss_sum += loss.item() * nonpad
            epoch_tokens += nonpad

            # TB step logs
            if global_step % args.log_every == 0:
                avg_loss = epoch_loss_sum / max(1, epoch_tokens)
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar("train/lr", lr_now, global_step)
                writer.add_scalar("train/tokens_per_step", nonpad, global_step)

        avg_loss_epoch = epoch_loss_sum / max(1, epoch_tokens)
        train_ppl = torch.exp(torch.tensor(avg_loss_epoch)).item()
        bleu, samples = evaluate_bleu_transformer(
            model, valid_loader, device,
            decode=args.decode, beam_size=args.beam_size, max_len=args.max_len,
            log_samples=args.log_samples, inv_src_vocab=inv_src, inv_tgt_vocab=inv_tgt
        )
        elapsed = time.time() - start_t

        print(f"[TRF] Epoch {epoch:02d} | loss={avg_loss_epoch:.4f} | ppl={train_ppl:.3f} | "
              f"valid_BLEU4={bleu*100:.2f} | time={elapsed:.1f}s | pos={args.pos_type} norm={args.norm_type}")

        # TB epoch logs
        writer.add_scalar("epoch/train_loss", avg_loss_epoch, epoch)
        writer.add_scalar("epoch/train_ppl", train_ppl, epoch)
        writer.add_scalar("epoch/valid_bleu4", bleu * 100.0, epoch)
        writer.add_scalar("epoch/epoch_time_sec", elapsed, epoch)

        if samples:
            writer.add_text("valid/samples", "\n\n".join(samples), epoch)

        if bleu > best_bleu:
            best_bleu = bleu
            save_checkpoint(
                args.save_path, model, optimizer=optim,
                meta={"epoch": epoch, "best_bleu": best_bleu, "args": vars(args)}
            )
            print(f"  -> saved best to {args.save_path} (BLEU4={best_bleu*100:.2f})")

    # TB hparams summary
    try:
        writer.add_hparams(
            hparam_dict={
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "num_layers": args.num_layers,
                "d_ff": args.d_ff,
                "dropout": args.dropout,
                "pos_type": args.pos_type,
                "norm_type": args.norm_type,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "label_smoothing": args.label_smoothing,
                "beam_size": args.beam_size,
            },
            metric_dict={"best_bleu4": best_bleu * 100.0},
        )
    except Exception:
        pass

    writer.close()


if __name__ == "__main__":
    main()
