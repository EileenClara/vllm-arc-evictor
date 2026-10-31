# Porting a KV-cache eviction policy from vLLM V0 to V1

V1 rewrote the engine. V0's pluggable `Evictor` base class is gone: eviction order is
now encoded in intrusive `prev`/`next` pointers on the block objects themselves, so you
cannot swap a policy by subclassing. You have to reimplement the `FreeKVCacheBlockQueue`
interface and keep its contract, while the policy kernel stays the same.

The paper's own implementation (`vllm-ARC`, branch `arc`) only touches `vllm/core/`.
`vllm/v1/` exists in that fork but was never modified. So everything below is unportable
knowledge: it is what actually breaks when the policy meets the V1 block pool.

Eleven traps, ordered by how much time they cost rather than by when they happened.

| # | Trap | Symptom | Silent? |
|---|---|---|---|
| 1 | Release path bypasses `add()` | hit rate collapses ~20 pp, `p` pins at max | no |
| 2 | Hashless blocks enter T1 | every evicted block has `hash=0`, ghost never fires | no |
| 3 | Two incompatible hash types | ghost hits stay at 0 after (2) is fixed | no |
| 4 | `_evict_from_T1` drops the hash | B1 stays empty under eviction | partial |
| 5 | `L1 > max_size` branch missing | `ref_cnt == 0` assertion, `POP bi=-1` | no |
| 6 | `remove()` treated as "upgrade" | evicts blocks that are still referenced | no |
| 7 | `_prune_ghosts` never runs | B2 unbounded, 520:1 hit ratio | partial |
| 8 | V1 reverses release order | nothing; 2.7 pp just disappears | **yes** |
| 9 | Debug clamp on block count | every cache tier reports the same number | **yes** |
| 10 | Hit rate measured wrong | 15% and 1.5% for the same policy | **yes** |
| 11 | Baseline was not the baseline | LRU 8.29% instead of 25.17% | **yes** |

Traps 8–11 never raise. They produce plausible numbers that are simply wrong, which is
why they cost more than the crashes.

---

## 1. The release path must call `add()`

**Symptom.** On a 3072-request QuALITY trace at 300k tokens, ARC scores **50.01%** against
LRU's **55.72%**. On the hotspot workload it is **20.90%** against **70.33%**. `p` climbs
monotonically to `max_size` and never returns.

**What I assumed.** That V1's eviction semantics were the cause. V1 treats "reusing a
hashed block" as an eviction, so evictions are far more frequent than in V0; the ghost
signal arrives in bursts, `delta = |B2| / |B1|` explodes, and `p` ratchets upward. Three
candidate fixes followed from that theory, and all three were tested:

- freeze `delta` at its batch-start value — `p` trajectory unchanged, to the block;
- check ghosts per block instead of per batch — hit rate 50.01% → 49.53%;
- bind the ghost check to actual eviction rather than every new block — instrumentation
  showed **96.7%** of checks already happened at a point where eviction was required.

All three were disproved. The theory was wrong.

**Root cause.** The adapter's `append_n()` — the release path for hashed blocks — called
the kernel's private `_add_to_T1` / `_add_to_T2` directly. `add()` is the **only** place
that executes `T1_active.remove(block_id)`. So every block that `evict()` had marked
in-flight stayed in `T1_active` forever:

```
L1 = |T1_table| + |T1_active| + |B1|   stayed permanently > max_size
→ evict() always took the forced _evict_from_T1() branch
→ _replace() was never reached
→ p became a dead variable
→ ARC degraded to unconditional T1-FIFO
```

**Fix.** Two lines at the top of `append_n()`:

```python
for b in blocks:
    bi = b.block_id
    self.evictor.T1_active.discard(bi)
    self.evictor.T2_active.discard(bi)
    # then _add_to_T1 / _add_to_T2 as before
```

**Verification.** Only ARC changed. LRU and DBL came out identical.

| metric | before | after |
|---|---|---|
| hotspot hit rate | 20.90% | 70.14% |
| QuALITY 300k hit rate | 50.01% | 56.48% |
| `T1_active` size | saturated (6242) | 1 |
| LRU / DBL @300k | 55.72 / 56.84 | 55.72 / 56.84 |

**Lesson.** The adapter owns an invariant: *every release must clear the active ledger*.
Any new release path that does not call `add()` re-introduces this. Traps 1 and 7 are the
same structural sin with different symptoms — before bypassing `add()`, count how many
invariants live inside it.

---

## 2. Hashless blocks must not enter T1/T2

**Symptom.** Every eviction logs `EVICT_T1 block=X hash=0`. B1 never receives a real
hash, so ghost lookup always misses.

**What I assumed.** An eviction-priority problem — that the wrong end of the queue was
being drained.

**Root cause.** V1's `free_blocks()` already splits the return path:

```python
blocks_without_hash -> free_block_queue.prepend_n(...)   # head
blocks_with_hash    -> free_block_queue.append_n(...)    # tail
```

The port pushed both into T1. Hashless blocks landed at the T1 head and were therefore
always the first thing evicted; hashed blocks sat behind them and were never evicted, so
B1 only ever saw zeros.

The real error was architectural. V0 kept `self._hashless_allocator` as a **separate
pool** — blocks that never accumulated 16 tokens (still generating, or abandoned by an
aborted request) had no content hash and never entered ARC's books at all. The port merged
two pools that V0 deliberately kept apart.

**Fix.** An independent `_hashless_pool`:

```python
prepend_n  -> _hashless_pool only
append_n   -> evictor only
popleft_n  -> drain _hashless_pool first, then evictor.evict()
```

This mirrors V0's `_allocate_block_id()`: try hashless first, then evicted.

**Verification.** 30 long documents against a 100-block cache:

```
EVICT non-zero hash: 12      EVICT hash=0: 0
```

---

## 3. Two incompatible hash types

**Symptom.** After trap 2, evicted blocks finally carry real hashes — and ghost hits are
still zero, even on a hand-built "evict this, then request it again" workload.

**Root cause.** `kv_cache_utils.py` defines two related types:

```python
BlockHash           = NewType("BlockHash", bytes)                 # bare hash
BlockHashWithGroupId = block_hash + group_id.to_bytes(4, "little") # + 4 bytes
```

The ghost lookup hashed `request.block_hashes[i]` (bare). The eviction path hashed
`block.block_hash` (with group id). The two values can never be equal, so B1 is searched
with a key that was never inserted.

**Fix.** Normalise both sides through `get_block_hash()`:

```python
self._hash(b) = hash(get_block_hash(b.block_hash))   # strips the 4-byte group id
```

**Verification.** A unit test covering B1 hit → `p` up → `_pending_t2` → `on_block_hashed`
→ release into T2, and B2 hit → `p` down: 10/10. On the integration workload `p` starts
moving for the first time.

**Lesson.** These two types are easy to hold in your head as "a hash is a hash". If you
instrument any hash-keyed structure in V1, log the type at both ends once.

---

## 4. `_evict_from_T1` does not write B1

**Symptom.** B1 stays empty even while eviction is plainly happening (9320 evictions
logged, B1 = 1 entry).

**Root cause.** `evict()` has two paths, and only one of them writes the ghost:

```python
if L1 == max_size:
    if len(T1_table) + len(T1_active) < max_size:
        B1.popitem()
        eid, ech = self._replace(content_hash)   # writes B1
    else:
        eid, ech = self._evict_from_T1()         # does NOT
```

**Fix.**

```python
else:
    eid, ech = self._evict_from_T1()
    self.B1[ech] = None        # record the evicted hash
```

This is conditional in a way that hides it: under a large cache the `_replace` branch
dominates and B1 looks healthy. It only appears when T1 is physically full — exactly the
regime where the ghost matters most.

---

## 5. The `L1 > max_size` branch is missing

**Symptom.** `AssertionError: block.ref_cnt == 0` inside `get_new_blocks`, with
`POP bi=-1` in the log immediately before it.

**Root cause.** The dispatch handled `L1 == max_size` and `L1 < max_size` but not
`L1 > max_size`, which happens whenever B1 is large. `eid` kept its sentinel value `-1`,
and `self._blocks[-1]` is valid Python — it silently returns the *last* block in the
pool, which is very likely in use.

**Fix.** Complete the branch (prune B1, then `_replace`), and refuse to index on failure:

```python
if eid < 0:
    raise RuntimeError("evict() returned no block")
```

**Lesson.** Python's negative indexing converts a control-flow bug into memory corruption
with no stack trace pointing at the cause. Never index a pool with a value that can be
`-1`; assert first.

---

## 6. `remove()` is not "upgrade"

**Symptom.** ARC evicts blocks whose `ref_cnt > 0`.

**Root cause.** In V1, `remove(block_id)` means *"this block is now in use — take it out
of the free queue"*. It carries a hard guarantee: the block must not appear in any
structure the evictor can select from. The port called `evictor.remove()` (which performs
the T1→T2 promotion) but left the block sitting in `T1_table` / `T2_table`, so a later
`evict()` could pick it.

**Fix.** Unconditionally pop from both tables, and record the *intent* to promote in a
separate set that `append_n()` consumes on release:

```python
self.evictor.T1_table.pop(bi, None)
self.evictor.T2_table.pop(bi, None)
self._promote_on_release.add(bi)
```

**Lesson.** The three V1 verbs map to ARC semantics, but not one-to-one — `remove` is
"hit + take ownership", and the promotion has to survive until release.

---

## 7. `_prune_ghosts` never runs

**Symptom.** B2 ghost hits **54,634** against B1's **105** — a 520:1 ratio — and ARC
loses to LRU at 3125 blocks.

**Root cause.** `_prune_ghosts()` is called from inside `add()`. Bypassing `add()` (trap 1
again) means it never executes, so B2 grows without bound. Every hash ever evicted stays
matchable, so nearly any revisit counts as a ghost hit, T2 protection runs unchecked, and
short-term hot content gets pushed out.

**Fix.** `self.evictor._prune_ghosts()` at the end of `append_n()`.

**Verification.** B2 hits 54,634 → 27,290. B1:B2 returned to a plausible range.

A 520:1 ratio between two queues that are supposed to be symmetric is the tell. Any
statistic that extreme is a bug report, not a finding.

---

## 8. V1 reverses release order

**Symptom.** None. No exception, no assertion. 2.7 percentage points simply vanish.

**Root cause.** Within a release batch V1 inserts the tail block first, then appends the
cached blocks in reverse order. Replaying them in forward order scrambles recency: the
oldest block is recorded as the newest.

**Fix.** Preserve the order the caller gave you. Do not sort, do not reverse, and do not
"normalise" a batch.

**Lesson.** The most expensive bugs in this list are the ones that cannot fail loudly.
The only defence is a differential test — same trace, same policy, wrapper vs reference
implementation — and the discipline to treat a 2.7 pp gap as a bug rather than noise.

---

## 9. The debug clamp

**Symptom.** Every cache tier in a five-tier sweep reports the identical hit rate.

**Root cause.** A debug branch pinned the block count to 3125 to force eviction while
working out bug 1:

```python
num_blocks = min(num_blocks, 3125)   # DEBUG
```

The sweep passed cache size in *tokens*, so every tier computed a different
`gpu_memory_utilization` and then got clamped to the same block count. Five tiers, one
cache, five identical numbers that look like a flat trend.

**Fix.** Read the block count from the environment, and make the harness assert that the
requested and actual counts agree.

**Lesson.** Any sweep should log *both* the requested and the effective cache size and
fail when they differ. A constant is not a result.

---

## 10. The hit-rate metric was wrong, twice

**Symptom.** The same policy measures **15%** with `batch=16` and **1.5%** with `batch=1`.

**Root cause.** The instrumentation read `request.num_computed_tokens`, which is *this
request's progress*, not *how many tokens came from cache*. Real hits live in
`num_new_computed_tokens`.

The two regimes fail in opposite directions, which is what made it confusing:

- `batch=1`: at allocation, progress is 0, so `num_c = 0` — the 1.5% is pure noise;
- `batch=16` with chunked prefill: progress accumulates across steps, so `num_c > 0` on
  later steps — the 15% is a chunked-prefill artifact.

Both numbers were meaningless, and both looked stable enough to reason from.

**Fix.**

```python
num_c = num_new_computed_tokens // block_size
```

**Verification.** The same prompt three times in a row: requests 2 and 3 report 372
blocks fully hit.

**Lesson.** Under chunked prefill, "tokens already computed" is not "tokens served from
cache". Instrument from the same source `PrefixCacheStats` uses, or you are measuring the
scheduler.

---

## 11. Your baseline was not the baseline

**Symptom.** ARC beats LRU by **24.9 pp** — roughly 2.3× the paper's reported upper bound.

**Root cause.** Two separate problems, found only because the number was implausibly
large.

First, the "LRU" side was running through the new wrapper (`ARCBlockQueue` + a
hand-written `LRUEvictor`), not through stock vLLM. Second, that `LRUEvictor` lacked the
staleness check ARC's kernel happens to have:

```python
# wrong: block may already be back in the table with a new timestamp
if bi in self._table:
    ...

# right
while self._heap:
    la, bi = heapq.heappop(self._heap)
    if bi in self._table and self._table[bi].last_accessed == la:
        ...
```

Repeated release/realloc cycles leave stale `(last_accessed, block_id)` entries in the
heap. Without the check, a block that was *just released* — the most recently used one —
gets evicted as if it were the oldest. LRU fell from 24.54% to 8.29%.

**Fix.** The check above. The wrapper then reproduced stock vLLM to four decimal places:
24.54% vs 24.54%.

The honest delta is **+7.29 pp** at 3125 blocks over 3072 requests, inside the paper's
1.2–10.8 pp range.

**Lesson.** Before claiming a delta, prove your baseline reproduces the reference
implementation. Two independent paths agreeing on the same number is the cheapest possible
check, and it is the one that would have caught both bugs.

---

## Checklist for the next port

1. Reimplement `popleft_n` / `remove` / `prepend_n` / `append_n` / `get_all_free_blocks`.
   Do not subclass — there is nothing to subclass.
2. Route **every** release through `add()`, or re-derive each invariant that lives inside
   it and clear it explicitly.
3. Keep hashless blocks in their own pool; they have no identity and must not be tracked.
4. Normalise hashes at the boundary. Log the type once.
5. Audit every branch of `evict()` for ghost writes — especially the forced ones.
6. Treat `ref_cnt` as a hard precondition, never as an assumption; raise instead of
   indexing with `-1`.
7. Make the sweep assert requested-vs-effective cache size.
8. Instrument hit rate from `num_new_computed_tokens`.
9. Diff your policy against a reference implementation on the same trace before believing
   any number.
10. When a statistic is 500:1, suspect the code before writing the finding.

---

## What this cost

| phase | wall clock | outcome |
|---|---|---|
| first working eviction | ~2 days | ARC runs, ghost never fires |
| ghost closure | ~2 days | B1 works, B2 unverified |
| p saturation investigation | ~5 days | three hypotheses, all disproved |
| root cause (trap 1) | ~1 hour after the right log | 20 pp recovered |
| baseline audit (trap 11) | ~1 day | 24.9 pp → 7.3 pp, honest |

The five days spent disproving three plausible theories were not waste — they are the
reason the answer is trustworthy. But the ordering matters: the log that settled it was a
single `T1_active` size print, and printing it on day one would have skipped four days of
theory.
