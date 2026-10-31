#!/usr/bin/env python3
"""Rebuild the block-level QuALITY trace that the offline sweeps consume.

The trace is a derived artifact: 3072 QuALITY requests, each expanded into the
content-hashed 16-token KV blocks its prompt would occupy in a real prefix-cache
run. Total block releases for the shipped experiments were 1,124,980 across all
five cache tiers.

It is not committed, for two reasons: it is a redistribution of a dataset we do
not own, and a binary blob hides the sampling decisions that matter. This script
regenerates it from the public corpus instead.

    python sim/make_trace.py --quality-json quality.json --model <hf-model-dir> \
        --out dataset/trace_quality.npz

Input format: a JSON list of {"article": str, "question": str}. The official
QuALITY release (arXiv:2112.11945) provides both fields.

Output (npz):
    flat     int64[total_blocks]   concatenated block ids, request after request
    offsets  int64[n_requests]     start of request i in `flat`
    lens     int64[n_requests]     number of blocks in request i

Block ids are stable: the first 16 bytes of sha1 over the block's token ids,
interpreted as a signed int64. Identical token spans collide by design — that is
exactly the prefix reuse the cache is supposed to exploit.

NOTE. This is a reconstruction from the experiment description (windowed Zipf,
512-request windows, exponent 1.0, batch=1), not a byte-for-byte recovery of the
original file: the original was produced on the experiment server and is not in
this repository. Absolute hit rates will therefore differ slightly from the
tables in the README; the *ranking* of policies is what the sweeps reproduce.
Run `--selftest` to verify the format end to end without any data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BLOCK_TOKENS = 16


def block_ids_for_tokens(token_ids: list[int], block_tokens: int = BLOCK_TOKENS) -> list[int]:
    """Split a token stream into fixed-size blocks and hash each one."""
    out = []
    for start in range(0, len(token_ids) - block_tokens + 1, block_tokens):
        chunk = np.asarray(token_ids[start:start + block_tokens], dtype=np.int64)
        digest = hashlib.sha1(chunk.tobytes()).digest()[:8]
        out.append(int.from_bytes(digest, "little", signed=True))
    return out  # trailing partial block carries no hash identity, as in vLLM


def zipf_windows(n_requests: int, n_docs: int, window: int, exponent: float,
                 seed: int) -> list[int]:
    """Draw document indices with the paper's windowed Zipf pattern."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_docs + 1, dtype=np.float64)
    weights = ranks ** (-exponent)
    weights /= weights.sum()
    picks = []
    while len(picks) < n_requests:
        picks.extend(rng.choice(n_docs, size=min(window, n_requests - len(picks)),
                                p=weights, replace=True).tolist())
    return picks


def build(corpus: list[dict], tokenizer, n_requests: int, window: int,
          exponent: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prompts = [f"{c['article']}\n\nQuestion: {c['question']}\nAnswer:" for c in corpus]
    picks = zipf_windows(n_requests, len(prompts), window, exponent, seed)

    flat: list[int] = []
    offsets: list[int] = []
    lens: list[int] = []
    for doc in picks:
        ids = tokenizer.encode(prompts[doc], add_special_tokens=False)
        blocks = block_ids_for_tokens(ids)
        offsets.append(len(flat))
        lens.append(len(blocks))
        flat.extend(blocks)
    return np.asarray(flat, dtype=np.int64), np.asarray(offsets, dtype=np.int64), np.asarray(lens, dtype=np.int64)


def selftest() -> int:
    class WhitespaceTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [abs(hash(w)) % 50000 for w in text.split()]

    corpus = [{"article": " ".join(f"w{i}" for i in range(200)), "question": "q?"} for _ in range(5)]
    flat, offsets, lens = build(corpus, WhitespaceTokenizer(), 20, 5, 1.0, 0)
    assert len(offsets) == len(lens) == 20
    assert offsets[0] == 0
    assert offsets[-1] + lens[-1] == len(flat)
    assert (lens > 0).all()
    reused = len(flat) - len(set(flat.tolist()))
    print(f"selftest OK  requests={len(lens)} blocks={len(flat)} reused_block_refs={reused}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quality-json", type=Path, help='JSON list of {"article", "question"}')
    ap.add_argument("--model", help="tokenizer name or local path")
    ap.add_argument("--out", type=Path, default=Path("dataset/trace_quality.npz"))
    ap.add_argument("--n-requests", type=int, default=3072)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--zipf-exponent", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.quality_json or not args.model:
        ap.error("--quality-json and --model are required unless --selftest is given")

    corpus = json.loads(args.quality_json.read_text(encoding="utf-8"))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    flat, offsets, lens = build(corpus, tokenizer, args.n_requests, args.window,
                                args.zipf_exponent, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, flat=flat, offsets=offsets, lens=lens)
    print(f"{args.out}: requests={len(lens)} blocks={len(flat)} "
          f"mean_blocks_per_request={lens.mean():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
