"""Offline block-level cache simulator for the QuALITY trace.

Fidelity strategy
-----------------
The POLICY layer is the byte-identical `customized_evictor.py` that ran on the
server (md5 4b65a79fdf7894c39b4eda88dfbf798e).  Only the "shim" layer -- the
part that vLLM's block_pool drives -- is reimplemented here, ported method by
method from arc_queue_full.py (ARCBlockQueue).  So ARC's behaviour does not
depend on anyone re-deriving ARC.

Request lifecycle, taken from vllm_kv_cache_manager.py::allocate_slots:
  1. get_computed_blocks()        -> longest prefix of block_hashes present
  2. allocate_new_computed_blocks -> touch each hit block (remove from free)
  3. CALL log                     <- arc_dbg.log line 484 is emitted HERE
  4. allocate_new_blocks()        -> popleft_n: hashless pool first, then evict
  5. cache_blocks()               -> check_ghost_hits + on_block_hashed
  6. free()                       -> append_n (hashed) / prepend_n (hashless)

The CALL log is deliberately NOT re-emitted (2.4M lines); counters are kept in
memory.  The evictor's own EVICT_T1/EVICT_T2 writes are intercepted and counted.
"""

import builtins
import os
import sys
import time
from collections import Counter, deque

import numpy as np

# ARC_P_CAP_FRAC=1.0 in run_b2decouple.sh -- must match the run being validated.
os.environ.setdefault('ARC_P_CAP_FRAC', '1.0')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

COUNTS = Counter()


class _CountingHandle:
    """Swallows the evictor's per-eviction `open(...).write(...).close()`."""

    __slots__ = ()

    def write(self, s):
        if s.startswith('EVICT_T1'):
            COUNTS['EVICT_T1'] += 1
        elif s.startswith('EVICT_T2'):
            COUNTS['EVICT_T2'] += 1
        return len(s)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_HANDLE = _CountingHandle()
_real_open = builtins.open


def _patched_open(file, *a, **kw):
    if isinstance(file, str) and file.startswith('/tmp/'):
        return _HANDLE
    return _real_open(file, *a, **kw)


builtins.open = _patched_open

from vllm.core.customized_evictor import CustomizedARCEvictor  # noqa: E402


class Shim:
    """Port of ARCBlockQueue's block-pool-facing methods (arc_queue_full.py)."""

    def __init__(self, max_size, p_freeze=None, evictor_cls=None,
                 evictor_kwargs=None):
        self.max_size = max_size
        # Default is the validated ARC path, unchanged.  evictor_cls exists so
        # DBL can be driven by this same shim: DBLEvictorAdapter already maps
        # the ARC interface onto DBL internals (T1_table->A1in, T2_table->Am,
        # both _add_to_T* routed through add() so ghost promotion still fires,
        # _prune_ghosts a no-op because DBL's ghosts are self-pruning deques).
        # The shim's only other ARC-specific touch is T1_active/T2_active, and
        # it uses those solely as .discard() targets on release -- the adapter
        # returning a fresh empty set each call is therefore harmless, because
        # nothing here ever READS them to make a decision.
        self.evictor = (evictor_cls or CustomizedARCEvictor)(
            max_size=max_size, **(evictor_kwargs or {}))
        self.p_freeze = p_freeze
        if p_freeze is not None:
            self.evictor.p = p_freeze
        self.num_free_blocks = max_size
        self._hashless_pool = deque()
        self._promote_on_release = {}
        self._pending_t2 = set()
        self._pending_t2_src = {}
        self._release_seq = 0
        # content id -> physical block id, for blocks currently in the prefix cache
        self.content_to_block = {}
        self.block_content = {}
        self.block_ntok = {}

    # ---- allocation ----------------------------------------------------
    def popleft_one(self):
        if self.num_free_blocks == 0:
            return None
        if self._hashless_pool:
            bi = self._hashless_pool.popleft()
            COUNTS['serve_hashless'] += 1
            self.num_free_blocks -= 1
            return bi
        bi, ch = self.evictor.evict(content_hash=None)
        if ch is not None and self.content_to_block.get(ch) == bi:
            del self.content_to_block[ch]
        self.block_content.pop(bi, None)
        self.num_free_blocks -= 1
        return bi

    # ---- hit / release -------------------------------------------------
    def remove(self, bi):
        was_t1 = bi in self.evictor.T1_table
        was_t2 = bi in self.evictor.T2_table
        # Port of arc_queue_full.py:142-160.  The real queue branches here, and
        # the branch is load-bearing for DBL:
        #
        #   if hasattr(self.evictor, 'A1in_free_table'):
        #       try: self.evictor.remove(bi)
        #       except ValueError: pass
        #   else:
        #       T1_table.pop / T2_table.pop
        #
        # DBL's own evict() does NOT record a ghost -- both append lines in
        # _evict_from_A1in/_evict_from_Am are commented out -- so `remove()` is
        # the ONLY writer of A1out/Amout, and `add()` is the only reader that
        # promotes to Am.  Popping DBL's tables directly, as the ARC path does,
        # therefore severs the promotion chain entirely: nothing is ever
        # ghosted, nothing is ever promoted, and DBL degenerates.  (It did --
        # silently, to 415,885 hits, which reads as a plausible number.)
        if hasattr(self.evictor, 'A1in_free_table'):
            try:
                self.evictor.remove(bi)
            except ValueError:
                pass
        else:
            self.evictor.T1_table.pop(bi, None)
            self.evictor.T2_table.pop(bi, None)
        if was_t1 or was_t2:
            self._promote_on_release[bi] = 'T1HIT' if was_t1 else 'T2HIT'
            COUNTS['REMOVE'] += 1
        self.num_free_blocks -= 1

    def _make_meta(self, bi, now):
        from vllm.core.evictor import BlockMetaData
        return BlockMetaData(self.block_content[bi], self.block_ntok[bi], now)

    def _release(self, bi):
        self._release_seq += 1
        self.evictor.T1_active.discard(bi)
        self.evictor.T2_active.discard(bi)
        if bi in self._promote_on_release:
            src = self._promote_on_release.pop(bi)
            self.evictor._add_to_T2(bi, self._make_meta(bi, self._release_seq))
            COUNTS['ADD_T2'] += 1
            COUNTS['ADD_T2_' + src] += 1
        else:
            self.evictor._add_to_T1(bi, self._make_meta(bi, self._release_seq))
            COUNTS['ADD_T1'] += 1
        self.num_free_blocks += 1

    def prepend_hashless(self, bi):
        self._hashless_pool.append(bi)
        self.block_content.pop(bi, None)
        COUNTS['ADD_HASHLESS'] += 1
        self.num_free_blocks += 1

    # ---- ghost ---------------------------------------------------------
    def check_ghost_hits(self, h):
        """p_freeze is not None -> frozen-p oracle (ARC-p*): the controller is
        pinned at a constant, but the ghost bookkeeping that gives a B1/B2 hit
        its admission ticket is left fully intact.  That isolates the value of
        *self-adaptation* from the value of the ghost lists themselves."""
        ev = self.evictor
        frozen = self.p_freeze is not None
        if h in ev.B1:
            if not frozen:
                d = max(1, len(ev.B2) // max(1, len(ev.B1)))
                ev.p = min(ev.p + d, ev.p_cap)
            del ev.B1[h]
            self._pending_t2.add(h)
            self._pending_t2_src[h] = 'B1'
            COUNTS['GHOST_B1'] += 1
        elif h in ev.B2:
            if not frozen:
                d = max(1, len(ev.B1) // max(1, len(ev.B2)))
                ev.p = max(ev.p - d, 0)
            del ev.B2[h]
            self._pending_t2.add(h)
            self._pending_t2_src[h] = 'B2'
            COUNTS['GHOST_B2'] += 1

    def on_block_hashed(self, h, bi):
        if h in self._pending_t2:
            src = self._pending_t2_src.pop(h, 'B1')
            self._pending_t2.discard(h)
            self._promote_on_release[bi] = src
            COUNTS['CONSUME'] += 1


def run(trace_path, max_size, n_requests=None, verbose=True, snapshot=None,
        trace_out=None, p_freeze=None, split_at=None, evictor_cls=None,
        evictor_kwargs=None):
    z = np.load(trace_path)
    flat, offsets, lens = z['flat'], z['offsets'], z['lens']
    n = len(offsets) - 1
    if n_requests:
        n = min(n, n_requests)
    if split_at is None:
        split_at = n // 2

    shim = Shim(max_size, p_freeze=p_freeze, evictor_cls=evictor_cls,
                evictor_kwargs=evictor_kwargs)
    # block_id 0 is vLLM V1's NULL block -- it is never in the free queue, so the
    # usable pool is 1..max_size and holds exactly max_size blocks.  Confirmed
    # from arc_dbg.log: max(T1+T2+T1a+T2a) == 6249 across the whole run.
    for bi in range(1, max_size + 1):
        shim._hashless_pool.append(bi)

    BS = 16
    t0 = time.time()
    snap = _real_open(trace_out, 'w') if trace_out else None
    for r in range(n):
        a, b = int(offsets[r]), int(offsets[r + 1])
        blocks = flat[a:b]
        nfull = len(blocks)
        # ceil(L/BS) is WRONG here: max_tokens=1 means the prefill finishes the
        # sequence, so no decode-step block is ever allocated.  A request with
        # L%16==0 therefore has NO hashless tail block at all.  Verified against
        # ghost.log: ADD_HASHLESS 2922 = 3072 - #{L%16==0} = 3072 - 150.
        has_tail = (int(lens[r]) % BS) != 0

        # 1+2. longest present prefix, then touch the hits
        k = 0
        while k < nfull:
            bi = shim.content_to_block.get(int(blocks[k]))
            if bi is None:
                break
            shim.remove(bi)
            k += 1
        COUNTS['hit_blocks'] += k
        COUNTS['total_blocks'] += nfull
        # split-half bookkeeping: lets a sweep pick p* on the first half and
        # score it on the second, so a frozen-p win is not an oracle artifact.
        if split_at is not None:
            h = 'h1' if r < split_at else 'h2'
            COUNTS['hit_' + h] += k
            COUNTS['blk_' + h] += nfull

        # 3. CALL log point -- counters/side effects are what matter, not the line
        if snap is not None:
            ev = shim.evictor
            snap.write('%d %d %d %d %d %d %d %d %d %d\n' % (
                r, nfull, k, ev.p, len(ev.T1_table), len(ev.T2_table),
                len(ev.B1), len(ev.B2), len(ev.T1_active), len(ev.T2_active)))

        # 4. allocate the misses, then the partial (hashless) tail block
        for j in range(k, nfull):
            bi = shim.popleft_one()
            if bi is None:
                raise RuntimeError('out of free blocks at req %d j %d' % (r, j))
            c = int(blocks[j])
            shim.content_to_block[c] = bi
            shim.block_content[bi] = c
            shim.block_ntok[bi] = (j + 1) * BS
        tail = shim.popleft_one() if has_tail else None
        if has_tail and tail is None:
            raise RuntimeError('out of free blocks (tail) at req %d' % r)

        # 5. ghost checks for the newly hashed blocks
        for j in range(k, nfull):
            c = int(blocks[j])
            shim.check_ghost_hits(c)
            shim.on_block_hashed(c, shim.content_to_block[c])

        # 6. free.  ORDER IS NOT PREFIX ORDER -- ghost.log line 1 is
        # `ADD_HASHLESS block=373` followed by `ADD_T1 block=372,371,...,1`, i.e.
        # the tail is prepended FIRST and the hashed blocks are handed to
        # append_n in REVERSE (deepest block first).  Because _release_seq is the
        # heap's primary key, this is what protects the shared prefix ROOT
        # (released last -> largest seq -> evicted last) and evicts the deepest,
        # most request-specific blocks first.  Releasing in forward order instead
        # inverts the whole eviction priority and costs ~2.7% hit rate.
        if has_tail:
            shim.prepend_hashless(tail)          # ArcQueue.prepend_n runs first
        for j in range(nfull - 1, -1, -1):
            shim._release(shim.content_to_block[int(blocks[j])])
        # ARCBlockQueue.append_n ends with one _prune_ghosts() per request --
        # without it B1/B2 grow unbounded and the delta formula goes runaway.
        shim.evictor._prune_ghosts()

        if verbose and (r + 1) % 500 == 0:
            print('  %4d/%d  %.0fs  hr=%.4f' % (r + 1, n, time.time() - t0,
                  COUNTS['hit_blocks'] / max(1, COUNTS['total_blocks'])), flush=True)

    if snap is not None:
        snap.close()
    return shim


def report(shim, label='ARC'):
    ev = shim.evictor
    hr = COUNTS['hit_blocks'] / max(1, COUNTS['total_blocks'])
    print()
    print('=' * 66)
    print('%s   max_size=%d' % (label, shim.max_size))
    print('=' * 66)
    # NOTE on the "server" column: the 0.4333 that used to be quoted for
    # hit_rate is a TOKEN-level ratio, sum(hit_tokens)/sum(total_tokens) over
    # arc_dbg.log -- and that sum double-counts the 17 requests that were split
    # by chunked prefill (max_num_batched_tokens=8192), inflating the
    # denominator by ~8,900 blocks.  The block-level truth is 491317/1124980 =
    # 0.43673.  Compare block counts, not the token ratio.
    rows = [
        ('hit_rate(blk)',  '%.5f' % hr,                 '0.43673'),
        ('B1_hit',         str(COUNTS['GHOST_B1']),     '49286'),
        ('B2_hit',         str(COUNTS['GHOST_B2']),     '28684'),
        ('EVICT_T1',       str(COUNTS['EVICT_T1']),     '533688'),
        ('EVICT_T2',       str(COUNTS['EVICT_T2']),     '93727'),
        ('p_final',        str(ev.p),                   '1224'),
        ('B2_final',       str(len(ev.B2)),             '1226'),
        ('ADD_HASHLESS',   str(COUNTS['ADD_HASHLESS']), '2922'),
        ('ADD_T1',         str(COUNTS['ADD_T1']),       '555693'),
        ('ADD_T2',         str(COUNTS['ADD_T2']),       '569287'),
        ('T2_T1HIT',       str(COUNTS['ADD_T2_T1HIT']), '20781'),
        ('T2_T2HIT',       str(COUNTS['ADD_T2_T2HIT']), '470536'),
        ('REMOVE(hits)',   str(COUNTS['REMOVE']),       '491317'),
    ]
    print('   %-12s %-14s %-14s' % ('metric', 'sim', 'server'))
    for name, got, want in rows:
        ok = 'OK ' if got == want else '   '
        print('%s %-12s %-14s %-14s' % (ok, name, got, want))
    print()
    for k in sorted(COUNTS):
        print('  %-16s %d' % (k, COUNTS[k]))
    print('  %-16s %d' % ('distinct_contents', len(set(shim.block_content.values()))))


if __name__ == '__main__':
    trace = os.path.join(HERE, '..', 'dataset', 'trace_quality.npz')
    t = time.time()
    shim = run(trace, max_size=6249, trace_out=os.path.join(HERE, 'sim_snap.txt'))
    report(shim)
    print('\nelapsed %.0fs' % (time.time() - t))
