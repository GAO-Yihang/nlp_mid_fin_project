from collections import Counter
from typing import List, Sequence, Tuple
import math

def _ngrams(tokens: Sequence, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def bleu4_corpus(references: List[List], hypotheses: List[List], max_n: int = 4) -> float:
    """
    references: list of reference token lists
    hypotheses: list of hypothesis token lists
    Returns BLEU-4 in [0,1].
    Notes:
      - single reference per hypothesis (your dataset is 1-to-1)
      - includes brevity penalty
    """
    assert len(references) == len(hypotheses) and len(references) > 0

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n

    ref_len_sum = 0
    hyp_len_sum = 0

    for ref, hyp in zip(references, hypotheses):
        ref_len_sum += len(ref)
        hyp_len_sum += len(hyp)

        for n in range(1, max_n + 1):
            hyp_ng = _ngrams(hyp, n)
            ref_ng = _ngrams(ref, n)

            total_counts[n-1] += sum(hyp_ng.values())
            for ng, c in hyp_ng.items():
                clipped_counts[n-1] += min(c, ref_ng.get(ng, 0))

    precisions = []
    for n in range(1, max_n + 1):
        if total_counts[n-1] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n-1] / total_counts[n-1])

    # If any precision is zero, BLEU is zero (common convention)
    if any(p == 0.0 for p in precisions):
        return 0.0

    # geometric mean
    log_p = sum(math.log(p) for p in precisions) / max_n
    geo_mean = math.exp(log_p)

    # brevity penalty
    if hyp_len_sum == 0:
        return 0.0
    bp = 1.0 if hyp_len_sum > ref_len_sum else math.exp(1.0 - ref_len_sum / hyp_len_sum)

    return bp * geo_mean
