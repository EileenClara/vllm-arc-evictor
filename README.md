# vllm-arc-evictor

ARC (Adaptive Replacement Cache) eviction policy for the vLLM V1 KV cache.

A faithful reimplementation of the ARC eviction policy from
[Shen et al., "Recency/Frequency Adaptive KV Caching for Large Language Model Serving" (arXiv:2606.21238)](https://arxiv.org/abs/2606.21238),
adapted to vLLM's V1 engine together with a small adapter layer that plugs the policy into the
V1 block-pool free queue. ARC needs **no workload parameter** — it self-tunes between
LRU and LFU via a single adaptive value `p`.

## What's in this repo

| File | Role | Deploy to |
|---|---|---|
| `customized_evictor_full_dbl.py` | L2 kernel: `CustomizedARCEvictor` + `CustomizedDBLEvictor` + `DBLEvictorAdapter` (three-in-one) | `vllm/core/customized_evictor.py` |
| `arc_queue_full.py` | L3 adapter: `ARCBlockQueue`, an ARC/LRU/DBL-switchable shell | the V1 free-block queue (`FreeKVCacheBlockQueue`) |
| `customized_evictor_full.py` | L2 kernel, ARC-only (no DBL) | `vllm/core/customized_evictor.py` |

`figures/` contains the headline results (hit rate vs cache size, ARC−LRU advantage,
and the V0/V1 `p` trajectory).

## ARC in one paragraph

ARC keeps four structures: two real caches **T1** (blocks seen once — recency) and **T2**
(blocks seen ≥2 times — frequency), plus two ghost lists **B1** / **B2** that remember only the
hashes of evicted blocks. A single adaptive parameter `p` (the target size of T1) moves on every
ghost hit: a B1 hit says "recency mattered" and raises `p`; a B2 hit says "frequency mattered" and
lowers `p`. This lets the cache self-tune between pure LRU (`p` = max) and pure LFU (`p` = 0)
without ever being told the access pattern.

## Design (L2 kernel + L3 adapter)

* **L2 kernel** (`CustomizedARCEvictor`) implements the four-queue bookkeeping, ghost promotion,
  and `_replace` (the only place `p` is read). Eviction order uses a `heapq` with lazy deletion:
  each block lives in a dict (`T1_table` / `T2_table`) for O(1) membership **and** a min-heap
  (`T1_queue` / `T2_queue`) keyed by `(last_accessed, -num_hashed_tokens, block_id, content_hash)`
  for O(log n) eviction.
* **L3 adapter** (`ARCBlockQueue`) implements the V1 `FreeKVCacheBlockQueue` interface
  (`popleft_n` / `remove` / `prepend_n` / `append_n` / `get_all_free_blocks`), delegating eviction
  decisions to the kernel. Blocks that have no hash identity (fewer than 16 tokens, i.e. no prefix
  reuse) are kept in a separate FIFO pool and never enter ARC bookkeeping.
* **Two-ledger bookkeeping**: the kernel tracks "in-flight" blocks in `T1_active` / `T2_active`
  (handed out by `evict()`, not yet released) separately from "evictable" blocks in
  `T1_table` / `T2_table`. A block's only legal path from in-flight back to evictable is `add()`.

### The bug this repo exists to document

The adapter's `append_n()` — the release path for hashed blocks — called the kernel's private
helpers `_add_to_T1` / `_add_to_T2` directly, **bypassing `add()`**, which is the *only* caller of
`T1_active.remove()`. Every block marked `T1_active` by `evict()` therefore never left the active
set; `T1_active` saturated to `max_size`; `_replace` short-circuited; and ARC silently degraded to
unconditional T1-FIFO. Hit rate collapsed ~50 percentage points (20.9% → 70.1% on the hotspot
workload) and TTFT rose with it. Two `discard()` lines in `append_n()` fix it.

This is documented here because it is easy to re-introduce when adapting the kernel to a new engine
version: **the adapter must always clear the active set on release.**

## Results

Reproduced on DeepSeek-R1-Distill-Qwen-1.5B, 16 tokens/block, two workloads:

* **QuALITY** (N ≈ 3 090 requests) — paper's workload, diverse long documents.
* **hotspot** (WildChat, N = 12 000) — shared system prefix, high eviction pressure.

Block hit rate (%) vs cache size (k tokens):

| workload | tier | ARC | LRU | DBL | ARC−LRU |
|---|---|---|---|---|---|
| QuALITY | 50k | 35.12 | 25.17 | 32.83 | **+9.95** |
| QuALITY | 100k | 43.33 | 36.66 | 41.50 | +6.67 |
| QuALITY | 150k | 48.17 | 43.72 | 47.36 | +4.45 |
| QuALITY | 200k | 51.65 | 48.61 | 51.09 | +3.04 |
| QuALITY | 300k | 56.48 | 55.72 | 56.84 | +0.76 |
| hotspot | 50k | 70.03 | 70.26 | 70.24 | −0.23 |
| hotspot | 100k | 70.14 | 70.33 | 70.29 | −0.19 |
| hotspot | 300k | 70.23 | 70.40 | 70.40 | −0.17 |

Two takeaways that reproduce the paper's Figure 3:

1. On QuALITY, **ARC beats LRU at every tier**, and the advantage narrows as the cache grows
   (+9.95 pp @ 50k → +0.76 pp @ 300k) — exactly the "adaptive advantage shrinks with capacity"
   shape the paper reports.
2. On the shared-prefix hotspot workload, ARC ≈ LRU ≈ DBL: when the working set already fits and
   eviction is dominated by a single shared prefix, there is little for a replacement policy to do.

## Integration

1. Copy the kernel to `vllm/core/customized_evictor.py` — use
   `customized_evictor_full_dbl.py` for ARC+DBL, or `customized_evictor_full.py` for ARC-only.
2. Wire `arc_queue_full.py` into the V1 block pool as the free-block queue.
3. Apply the `block_pool.py` changes (see "Known gaps").
4. Select the policy and cache size via environment variables:

   ```bash
   export VLLM_EVICTOR=ARC          # ARC | LRU | DBL  (default ARC)
   export VLLM_CACHE_TOKENS=100000  # cache budget in tokens
   export VLLM_CACHE_BLOCKS=6250    # = VLLM_CACHE_TOKENS / 16
   export ARC_P_CAP_FRAC=0.5        # soft-cap on p, as a fraction of max_size
   export VLLM_N_REQUESTS=12000     # stop after N requests (for sweeps)
   ```

## Known gaps

* The `block_pool.py` modifications (three lines that wire `VLLM_CACHE_BLOCKS` into the block
  count, so each sweep tier gets a *distinct* cache size rather than clamping to one) live only on
  the experiment server and are not yet mirrored here. Without them the cache-size sweep degenerates:
  every tier reports the same hit rate.
* DBL eviction uses the same `DBLEvictorAdapter` here as in the experiment; its tuning is not
  independently verified against the paper beyond the numbers above.

## License

Apache-2.0 (aligning with vLLM). See [LICENSE](LICENSE).
