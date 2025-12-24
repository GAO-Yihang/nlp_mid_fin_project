"""
Preprocessing for Zh->En NMT (train/valid/test) with:
- Data cleaning (NFKC, remove control chars, normalize spaces)
- Tokenization: zh=jieba, en=regex (keeps punctuation)
- Length filtering (train stricter; eval can be looser)
- Vocabulary building ONLY from train (no leakage)
- Encoding with <bos>/<eos>, OOV -> <unk>
- Outputs:
  out_dir/
    data.clean.jsonl
    data.tok.jsonl
    data.ids.jsonl
    (train only) vocab.src.json, vocab.tgt.json
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# -----------------------
# 1) basic cleaning utils
# -----------------------
CTRL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
MULTI_SPACE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = CTRL_CHARS.sub("", s)
    s = s.replace("\ufeff", "").replace("\u200b", "")
    s = MULTI_SPACE.sub(" ", s).strip()
    return s

# -----------------------
# 2) tokenization
# -----------------------
def tokenize_en(s: str) -> List[str]:
    """
    Tokenize English into words and punctuation.
    Example: "1989?" -> ["1989", "?"]
    """
    s = s.lower()
    return re.findall(r"[a-z0-9]+|[^\w\s]", s, flags=re.UNICODE)

def tokenize_zh_jieba(s: str) -> List[str]:
    """
    Jieba tokenization for Chinese.
    """
    import jieba
    return [t.strip() for t in jieba.lcut(s) if t.strip()]

# -----------------------
# 3) vocab + encoding
# -----------------------
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]

def build_vocab(counter: Counter, max_vocab: int = 50000, min_freq: int = 2) -> Dict[str, int]:
    """
    Build vocab from token frequency counter.
    Keeps SPECIALS at the beginning in fixed order.
    """
    # filter by min_freq
    words = [w for w, c in counter.items() if c >= min_freq]
    # sort by frequency desc, then lexicographically for stability
    words.sort(key=lambda w: (-counter[w], w))
    # cut to max_vocab
    words = words[: max(0, max_vocab - len(SPECIALS))]

    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for w in words:
        vocab[w] = len(vocab)
    return vocab

def encode(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab["<unk>"]
    return [vocab.get(t, unk) for t in tokens]

# -----------------------
# 4) IO helpers
# -----------------------
def read_jsonl(in_path: str) -> List[dict]:
    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(out_path: Path, rows: List[dict]) -> None:
    with open(out_path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_json(out_path: Path, obj: dict) -> None:
    with open(out_path, "w", encoding="utf-8") as w:
        json.dump(obj, w, ensure_ascii=False, indent=2)

def load_json(in_path: str) -> dict:
    with open(in_path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------
# 5) shared preprocessing steps (clean + tokenize + filter)
# -----------------------
def clean_and_tokenize(
    in_path: str,
    max_src_len: int,
    max_tgt_len: int,
    drop_if_empty: bool = True,
) -> Tuple[List[dict], List[dict]]:
    """
    Returns:
      clean_rows: [{"src": zh_clean, "tgt": en_clean, "index": idx}, ...]
      tok_rows:   [{"src_tok": [...], "tgt_tok": [...], "index": idx}, ...]  (filtered by length)
    """
    raw = read_jsonl(in_path)

    clean_rows: List[dict] = []
    tok_rows: List[dict] = []

    for obj in raw:
        zh = normalize_text(obj.get("zh", ""))
        en = normalize_text(obj.get("en", ""))
        idx = obj.get("index", None)

        if drop_if_empty and (not zh or not en):
            continue

        clean_rows.append({"src": zh, "tgt": en, "index": idx})

        src_tok = tokenize_zh_jieba(zh)
        tgt_tok = tokenize_en(en)

        # length filter
        if len(src_tok) > max_src_len or len(tgt_tok) > max_tgt_len:
            continue

        tok_rows.append({"src_tok": src_tok, "tgt_tok": tgt_tok, "index": idx})

    return clean_rows, tok_rows

# -----------------------
# 6) TRAIN preprocessing (build vocab + encode)
# -----------------------
def preprocess_train(
    in_path: str,
    out_dir: str,
    max_src_len: int = 80,
    max_tgt_len: int = 80,
    min_freq: int = 2,
    max_vocab: int = 50000,
) -> None:
    """
    Train-only preprocessing:
    - clean + tokenize + filter
    - build vocab from train tokens only
    - encode using this vocab
    - save clean/tok/ids and vocab files
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_rows, tok_rows = clean_and_tokenize(
        in_path=in_path,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )

    # save cleaned and tokenized
    write_jsonl(out_dir / "data.clean.jsonl", clean_rows)
    write_jsonl(out_dir / "data.tok.jsonl", tok_rows)

    # build vocab from train only
    src_cnt, tgt_cnt = Counter(), Counter()
    for r in tok_rows:
        src_cnt.update(r["src_tok"])
        tgt_cnt.update(r["tgt_tok"])

    src_vocab = build_vocab(src_cnt, max_vocab=max_vocab, min_freq=min_freq)
    tgt_vocab = build_vocab(tgt_cnt, max_vocab=max_vocab, min_freq=min_freq)

    save_json(out_dir / "vocab.src.json", src_vocab)
    save_json(out_dir / "vocab.tgt.json", tgt_vocab)

    # encode
    ids_rows = []
    for r in tok_rows:
        src_ids = [src_vocab["<bos>"]] + encode(r["src_tok"], src_vocab) + [src_vocab["<eos>"]]
        tgt_ids = [tgt_vocab["<bos>"]] + encode(r["tgt_tok"], tgt_vocab) + [tgt_vocab["<eos>"]]
        ids_rows.append({"src_ids": src_ids, "tgt_ids": tgt_ids, "index": r["index"]})

    write_jsonl(out_dir / "data.ids.jsonl", ids_rows)

    print(f"[TRAIN] Loaded {len(clean_rows)} cleaned pairs.")
    print(f"[TRAIN] Kept {len(tok_rows)} pairs after length filter (src<={max_src_len}, tgt<={max_tgt_len}).")
    print(f"[TRAIN] src_vocab size={len(src_vocab)} (min_freq={min_freq}, max_vocab={max_vocab})")
    print(f"[TRAIN] tgt_vocab size={len(tgt_vocab)} (min_freq={min_freq}, max_vocab={max_vocab})")
    print(f"[TRAIN] Saved to: {out_dir.resolve()}")

# -----------------------
# 7) EVAL preprocessing (encode with train vocab only)
# -----------------------
def preprocess_eval(
    in_path: str,
    out_dir: str,
    src_vocab_path: str,
    tgt_vocab_path: str,
    max_src_len: int = 120,
    max_tgt_len: int = 120,
) -> None:
    """
    Eval preprocessing for valid/test:
    - clean + tokenize + (looser) length filter
    - DO NOT build vocab
    - encode using train vocab
    - save clean/tok/ids (no vocab)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_vocab = load_json(src_vocab_path)
    tgt_vocab = load_json(tgt_vocab_path)

    clean_rows, tok_rows = clean_and_tokenize(
        in_path=in_path,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )

    write_jsonl(out_dir / "data.clean.jsonl", clean_rows)
    write_jsonl(out_dir / "data.tok.jsonl", tok_rows)

    ids_rows = []
    for r in tok_rows:
        src_ids = [src_vocab["<bos>"]] + encode(r["src_tok"], src_vocab) + [src_vocab["<eos>"]]
        tgt_ids = [tgt_vocab["<bos>"]] + encode(r["tgt_tok"], tgt_vocab) + [tgt_vocab["<eos>"]]
        ids_rows.append({"src_ids": src_ids, "tgt_ids": tgt_ids, "index": r["index"]})

    write_jsonl(out_dir / "data.ids.jsonl", ids_rows)

    print(f"[EVAL] Loaded {len(clean_rows)} cleaned pairs.")
    print(f"[EVAL] Kept {len(tok_rows)} pairs after length filter (src<={max_src_len}, tgt<={max_tgt_len}).")
    print(f"[EVAL] Using src_vocab={Path(src_vocab_path).resolve()}")
    print(f"[EVAL] Using tgt_vocab={Path(tgt_vocab_path).resolve()}")
    print(f"[EVAL] Saved to: {out_dir.resolve()}")

# -----------------------
# 8) Example usage
# -----------------------
if __name__ == "__main__":
    # Paths (edit these)
    train_path = "/data/250010229/phd_hw/AP0004/midfin/AP0004_Midterm&Final_translation_dataset_zh_en/train_100k.jsonl"
    valid_path = "/data/250010229/phd_hw/AP0004/midfin/AP0004_Midterm&Final_translation_dataset_zh_en/valid.jsonl"
    test_path  = "/data/250010229/phd_hw/AP0004/midfin/AP0004_Midterm&Final_translation_dataset_zh_en/test.jsonl"

    # Output dirs
    train_out = "/data/250010229/phd_hw/AP0004/midfin/data/train"
    valid_out = "/data/250010229/phd_hw/AP0004/midfin/data/valid"
    test_out  = "/data/250010229/phd_hw/AP0004/midfin/data/test"

    # 1) Train preprocessing (build vocab here)
    preprocess_train(
        in_path=train_path,
        out_dir=train_out,
        max_src_len=80,
        max_tgt_len=80,
        min_freq=2,
        max_vocab=50000,
    )

    # 2) Valid/Test preprocessing (encode using train vocab only)
    src_vocab_path = str(Path(train_out) / "vocab.src.json")
    tgt_vocab_path = str(Path(train_out) / "vocab.tgt.json")

    preprocess_eval(
        in_path=valid_path,
        out_dir=valid_out,
        src_vocab_path=src_vocab_path,
        tgt_vocab_path=tgt_vocab_path,
        max_src_len=120,
        max_tgt_len=120,
    )

    preprocess_eval(
        in_path=test_path,
        out_dir=test_out,
        src_vocab_path=src_vocab_path,
        tgt_vocab_path=tgt_vocab_path,
        max_src_len=200,
        max_tgt_len=200,
    )

