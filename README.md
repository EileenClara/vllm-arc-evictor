# vllm-arc-evictor

ARC (Adaptive Replacement Cache) eviction policy for the vLLM V1 KV cache, three follow-up
eviction policies built on what ARC taught us, and the offline simulator used to design them.

The starting point is a faithful reimplementation of the ARC eviction policy from
[Shen et al., "Recency/Frequency Adaptive KV Caching for Large Language Model Serving" (arXiv:2606.21238)](https://arxiv.org/abs/2606.21238),
adapted to vLLM's V1 engine together with a small adapter layer that plugs the policy into the
V1 block-pool free queue. ARC needs **no workload parameter** — it self-tunes between
LRU and LFU via a single adaptive value `p`.

On QuALITY that reproduction works: ARC beats LRU at every cache size. On two other axes it does
not, and the three follow-up policies below are that investigation.

## What's in this repo

| File | Role | Deploy to |
|---|---|---|
| `customized_evictor_full_dbl.py` | L2 kernel: `CustomizedARCEvictor` + `CustomizedDBLEvictor` + `DBLEvictorAdapter` (three-in-one). Holds improvement 1's `ARC_GHOST_CAP_B2` knob | `vllm/core/customized_evictor.py` |
| `customized_evictor_2q_ghost.py` | L2 kernel: `Customized2QGhostEvictor`, improvement 3 | `vllm/core/customized_evictor.py` |
| `arc_queue_full.py` | L3 adapter: `ARCBlockQueue`, an ARC/LRU/DBL/2Q+ghost-switchable shell | the V1 free-block queue (`FreeKVCacheBlockQueue`) |
| `customized_evictor_full.py` | L2 kernel, ARC-only (no DBL) | `vllm/core/customized_evictor.py` |
| `sim/policies.py` | Offline policies: `SingleQueue` (improvement 2's C0–C3 ladder) and `TwoQueue` (improvement 3) | — |
| `sim/arcsim.py` | Validated offline block-level simulator, drives the real ARC kernel | — |
| `sim/sweep_c.py`, `sim/sweep_c3c.py` | Improvement 2's ladder and the final C3 grid | — |
| `sim/sweep_2q.py`, `sim/sweep_2q_ghost.py` | Improvement 3's two steps, 2Q and 2Q + shared ghost | — |
| `sim/_verify_ghost_coupling.py` | Improvement 1's offline test, B2 starvation versus capacity | — |

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

## Three improvement attempts

Each one was run to test one claim about where ARC's advantage comes from. The first two came back
negative, and the third is what the two negatives compose into.

### 1. Decouple B2 — `ARC_GHOST_CAP_B2`

**Idea.** ARC bounds the frequency ghost by `|T2_table| + |T2_active| + |B2| <= max_size`, so when
`T2` fills the cache, `B2` is pruned to almost nothing and ARC forgets exactly the frequency
working set that a longer ghost would remember. Letting `B2` grow on its own should recover it.

**Result.** It makes things worse. On QuALITY at 100k the control ARC run scores 43.33%. Lifting the
B2 cap to 70 000 blocks drops it to 42.39%, and switching to the paper's original delta formula
drops it to 39.06%.

**What it proved.** B2's capacity and ARC's `p` adaptation are structurally incompatible. `p` is
corrected by the *ratio* of B1 to B2 hits, which is what compensates for the two ghost lists
receiving unequal traffic. Enlarging B2 alone breaks that compensation: B2 grows to ~56 000 while
B1 stays near 5 800, a 9.6:1 ratio, and `p` is driven to a rail with no healthy fixed point left.
Any policy that wants to exploit the frequency tail has to bypass `p` adaptation rather than
retune it. The knob and the offline test for the starvation hypothesis
(`sim/_verify_ghost_coupling.py`) are both here.

### 2. Single-queue scoring — the C0–C3 ladder

**Idea.** Drop the two-queue partition. Keep one queue and score every block explicitly, then add
one mechanism at a time and measure each increment: frequency evidence, then a ghost admission
ticket, then a recency-resonance term.

**Result.** On QuALITY at 100k, against a 1 124 980-block trace:

| rung | what it adds | hits | hit rate | Δ vs previous |
|---|---|---|---|---|
| C0 | plain LRU | 415 885 | 0.36968 | — |
| C1 | `+ beta·f(h)`, a saturating per-block frequency counter | 430 710 | 0.38286 | **+1.318 pp** |
| C2 | `+` ghost map, resurrection admits with `h ← 2` | 429 759 | 0.38201 | −0.085 pp |
| C3 | `+ gamma_y·g(r_y) + gamma_o·g(r_o)` | 429 759 | 0.38201 | 0.000 pp |

ARC scores 0.43664 and ARC with `p` frozen scores 0.44473 on the same trace, both far above the
ladder's ceiling.

**What it proved.** Explicit scoring buys the frequency term and nothing else — about 1.3 pp. The
ghost's *existence* and its *magnitude* add nothing on a single queue. ARC's real advantage is
therefore not in its scoring terms but in the two-queue partition structure itself, which the
ladder does not have. Keep the partition.

Two notes on reading these numbers. C1's windowed score mixing recency with a saturating frequency
counter is LRFU (Lee et al. 2001), so a C1 result is an LRFU result and not a ghost result. And
C2/C3 are gated behind a `tau` calibrated on a residency pilot, so `sim/sweep_c3c.py` scans the
full tau × gamma grid rather than a single operating point — two earlier C3 runs declared the
resonance term inert when in fact their `tau` sat below the residency distribution's p10 and the
term never switched on.

### 3. Two queues + one shared ghost — `Customized2QGhostEvictor`

**Idea.** Improvement 1 says bypass `p` adaptation. Improvement 2 says keep the partition. Combine
them: keep T1/T2 and the ghost's deferred promotion, remove `p` and the delta controller, and
collapse B1 and B2 into a single shared ghost so its capacity is no longer coupled to the queue
sizes.

**Result.** Both steps are in `sim/sweep_2q.py` and `sim/sweep_2q_ghost.py`.

| policy | hits | hit rate | vs LRU | vs ARC |
|---|---|---|---|---|
| C0 LRU | 415 885 | 0.36968 | — | −6.70 pp |
| 2Q, two queues, no ghost | 482 837 | 0.42920 | +5.95 pp | −0.74 pp |
| ARC, adaptive `p` | 491 216 | 0.43664 | +6.70 pp | — |
| ARC, `p` frozen at its best value | 500 311 | 0.44473 | +7.50 pp | +0.81 pp |
| **2Q + one shared ghost** | **505 087** | **0.44897** | **+7.93 pp** | **+1.23 pp** |

The two-queue structure alone already captures 89% of ARC's advantage over LRU. Adding one shared
ghost then clears ARC's frozen-`p` ceiling.

**What it showed.** ARC carries two penalties against this design: the delta controller costs
0.79 pp, and the B1/B2 complementary coupling that starves B2 costs 0.44 pp. The ceiling is not
ARC, and it is not the ghost either — it is a frequency queue plus a single shared ghost list.

The deployed evictor is `Customized2QGhostEvictor`. `k` is T1's floor: eviction takes from T1 while
`len(T1) > k` and from T2 once T1 has shrunk to `k`, so `k = 0` means T1 has no floor at all. The
validated configuration is `k = 0` with the ghost capped at `max_size`. Note that `k` is
semantically the same quantity as ARC's `p` — a fixed `k > 0` is a frozen `p`, and offline the hit
rate falls monotonically from 0.44897 at `k = 0` to 0.15 at `k = 6248`.

## Results

Reproduced on DeepSeek-R1-Distill-Qwen-1.5B, 16 tokens/block, two workloads:

* **QuALITY** (N ≈ 3 090 requests) — paper's workload, diverse long documents.
* **hotspot** (WildChat, N = 12 000) — shared system prefix, high eviction pressure.

Block hit rate (%) vs cache size (k tokens):

| workload | tier | 2Q+ghost | ARC | LRU | DBL |
|---|---|---|---|---|---|
| QuALITY | 50k | **36.15** | 35.12 | 25.17 | 32.83 |
| QuALITY | 100k | **44.74** | 43.33 | 36.66 | 41.50 |
| QuALITY | 150k | **49.43** | 48.17 | 43.72 | 47.36 |
| QuALITY | 200k | **52.90** | 51.65 | 48.61 | 51.09 |
| QuALITY | 300k | **57.76** | 56.48 | 55.72 | 56.84 |

On QuALITY, **ARC beats LRU at every tier**, and the advantage narrows as the cache grows
(+9.95 pp @ 50k → +0.76 pp @ 300k) — exactly the "adaptive advantage shrinks with capacity" shape
the paper reports. 2Q+ghost adds roughly another percentage point on top of ARC at every tier, and
its lead over LRU reaches 11 pp at 50k.

The hotspot workload tells the opposite story, and only at 100k do we have all four policies:

| tier | 2Q+ghost | ARC | LRU | DBL |
|---|---|---|---|---|
| hotspot 50k | — | 70.03 | 70.26 | 70.24 |
| hotspot 100k | **51.52** | 70.14 | 70.33 | 70.29 |
| hotspot 300k | — | 70.23 | 70.40 | 70.40 |

At 100k, ARC ≈ LRU ≈ DBL at ~70% while 2Q+ghost collapses 18.6 pp below LRU. This is a real
structural limit of `k = 0`, not an implementation bug — the eviction trace shows T1 receiving only
fresh misses while T2 takes 2.4x the traffic, T1 hitting on 2.8% of references, and 40.8% of all
evictions forced out of T2 because T1 was empty. Hotspot is short-hot-set recency reuse, where the
frequency ghost has nothing to do and the missing T1 floor hurts. Fixing it with a fixed `k > 0`
would cost QuALITY, so it is left as the open problem: an *adaptive* T1 floor, which is what ARC's
`p` is, minus the two penalties improvement 3 identified.

## Reproducing the offline numbers

The simulator is block-level, deterministic, and validated against the real V1 run to within 0.02%.

```bash
pip install numpy
# the kernel must be importable: arcsim.py drives the real CustomizedARCEvictor
cp customized_evictor_full_dbl.py <site-packages>/vllm/core/customized_evictor.py

# trace: sim/../dataset/trace_quality.npz, keys flat / offsets / lens,
# 3072 QuALITY requests expanded to block ids.  Not shipped here.
mkdir -p dataset && cp <your trace_quality.npz> dataset/

python sim/sweep_c.py                # improvement 2, the C0-C3 ladder
python sim/sweep_c3c.py              # improvement 2, full tau x gamma grid
python sim/sweep_2q.py               # improvement 3, step 1: two queues only
python sim/sweep_2q_ghost.py         # improvement 3, step 2: plus shared ghost
python sim/_verify_ghost_coupling.py # improvement 1, B2 starvation vs capacity
```

## Integration

1. Copy the kernel to `vllm/core/customized_evictor.py` — use
   `customized_evictor_full_dbl.py` for ARC+DBL, `customized_evictor_full.py` for ARC-only, and
   append `customized_evictor_2q_ghost.py` for the 2Q+ghost policy.
2. Wire `arc_queue_full.py` into the V1 block pool as the free-block queue.
3. Apply the `block_pool.py` changes (see "Known gaps").
4. Select the policy and cache size via environment variables:

   ```bash
   export VLLM_EVICTOR=ARC          # ARC | LRU | DBL | TWOQ_GHOST  (default ARC)
   export VLLM_CACHE_TOKENS=100000  # cache budget in tokens
   export VLLM_CACHE_BLOCKS=6250    # = VLLM_CACHE_TOKENS / 16
   export VLLM_N_REQUESTS=12000     # stop after N requests (for sweeps)
   ```

   ARC and DBL knobs:

   ```bash
   export ARC_P_CAP_FRAC=0.5        # soft-cap on p, as a fraction of max_size
   export ARC_GHOST_CAP_B2=70000    # improvement 1: decouple the B2 ghost cap from max_size
   export ARC_T2_FREQ_ENABLE=1      # optional T2-internal frequency counter, bypasses p
   export ARC_T2_FREQ_WINDOW=8
   export ARC_T2_FREQ_DECAY=0.5
   ```

   `VLLM_GHOST_LOG=<path>` makes 2Q+ghost append one line per eviction (`EVICT_T1` / `EVICT_T2`)
   for tracing. It is off by default, and it never feeds back into an eviction decision.

## Known gaps

* The `block_pool.py` modifications (three lines that wire `VLLM_CACHE_BLOCKS` into the block
  count, so each sweep tier gets a *distinct* cache size rather than clamping to one) live only on
  the experiment server and are not yet mirrored here. Without them the cache-size sweep degenerates:
  every tier reports the same hit rate.
* DBL eviction uses the same `DBLEvictorAdapter` here as in the experiment; its tuning is not
  independently verified against the paper beyond the numbers above.
* The offline simulator needs `dataset/trace_quality.npz`, which is not in this repo.
* The C0–C3 ladder was swept on QuALITY only. The hotspot workload has no offline trace, so
  improvement 3's collapse there rests on the online run and its eviction trace alone.

## License

Apache-2.0 (aligning with vLLM). See [LICENSE](LICENSE).
