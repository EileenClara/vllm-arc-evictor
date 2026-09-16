"""C0/C1/C2/C3 -- the single-queue ladder, run inside the validated simulator.

Ladder (the design's own defs):

  C0  LRU              anchor.  Cost 0.
  C1  {h}              window argmin(age_rank + beta*f(h)).  NO ghost structure
                       exists at all: a block that is evicted and later
                       re-accessed is indistinguishable from a brand-new one.
                       cost = one saturating counter per block.
  C2  {h + ticket}     ghost map exists; a ghost hit re-admits the content with
                       h <- 2 (the "ticket").  r_y/r_o are recorded on the
                       block, but gamma = 0 so they never enter the score.
                       cost = the ghost map.
  C3  full             + gamma_y*g(r_y) + gamma_o*g(r_o) in the score.

  Delta_h      = C1 - C0   what block-level frequency evidence is worth, zero
                           ghost overhead
  Delta_ticket = C2 - C1   what the ghost's *existence* (admission ticket) is
                           worth, with the resonance term switched off
  Delta_r      = C3 - C2   what the ghost's *magnitude* is worth on top

C2 must implement the ticket as literally "on resurrection, h <- 2".  The
original design routed resurrection through a `proven` flag, but `proven` was
dropped from the block state; spelling the ticket as an h-injection is what
keeps C2->C3 a one-variable change instead of smuggling in an admission-
semantics change as well.

Everything below the policy -- pool layout, hashless tail, allocation order,
release order -- is deliberately identical to arcsim.py's validated shim, so a
C-level number sits on the same footing as the ARC number.

Two mechanical facts inherited from validation, both easy to get wrong:
  * block_id 0 is vLLM V1's NULL block; the usable pool is 1..max_size (6249).
  * a request's hashed blocks reach the evictor in REVERSE prefix order, with
    the hashless tail prepended first.  Since the queue is ordered by release,
    this is what protects the shared prefix ROOT.  Releasing forward instead
    inverts recency for every policy here and silently costs ~2.7%.

Scoring.  age_norm in [0,1] is the candidate's rank inside the window (0 =
oldest, i.e. the pure-LRU victim), f(h) = min(h,H_CAP)/H_CAP, and likewise
g(r).  Lower S is evicted first, so beta/gamma read as "how many window-widths
is one unit of evidence worth".  window=1 collapses every C-level to LRU.

Prior art to keep in view when reading C1's number: a windowed per-block score
mixing recency with a saturating frequency counter is LRFU (Lee et al. 2001).
If C1 alone already matches ARC, that is an LRFU result, not a ghost result.
"""

import os
import sys
from collections import OrderedDict, deque
from itertools import islice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

BS = 16
H_CAP = 4.0             # h saturates at 4
R_CAP = 4.0             # g(r) saturates at r = 4
TICKET_H = 2.0          # C2: resurrection admits with h <- 2


class Counters(dict):
    def __missing__(self, k):
        return 0


class SingleQueue:
    """One recency-ordered queue of evictable cached blocks + a hashless pool."""

    def __init__(self, max_size, pol=1, window=1, beta=1.0,
                 gamma_y=0.0, gamma_o=0.0, tau=None, ghost_cap=None,
                 counters=None):
        self.max_size = max_size
        self.pol = pol                    # 0=LRU 1={h} 2=+ticket 3=full
        self.window = window
        self.beta = beta
        self.gamma_y = gamma_y
        self.gamma_o = gamma_o
        self.tau = float('inf') if tau is None else tau
        # `is not None`, not truthiness: ghost_cap=0 is a MEANINGFUL setting (no
        # ghost at all, which is what makes C2 collapse onto C1), and `if
        # ghost_cap` silently rewrote it to max_size -- i.e. the largest possible
        # ghost.  An invariant that cannot be expressed is worse than no
        # invariant, because the run still produces a plausible number.
        self.ghost_cap = max_size if ghost_cap is None else ghost_cap
        self.C = counters if counters is not None else Counters()

        self.order = OrderedDict()        # block_id -> content, OLDEST first
        self.content_to_block = {}
        self.hashless = deque(range(1, max_size + 1))
        self.seq = 0                      # release clock

        self.h = {}                       # block_id -> residency-local hit count
        self.blk_r = {}                   # block_id -> 案底 carried by this residency
        self.blk_gy = {}                  # block_id -> last eviction was young?
        # Residency duration = eviction_seq - birth_seq, where birth is set at
        # ADMISSION ONLY.  It must not be refreshed on release: refreshing it
        # measures idle age ("time since last use"), which is a different
        # quantity and makes tau nearly inert, because idle age is bimodal
        # (just-used vs long-abandoned) and any tau inside the gap splits
        # identically.  The design's G_y/G_o split is by residency duration.
        self.birth = {}                   # block_id -> seq at admission
        # Pre-baked score terms.  The victim scan runs ~6e5 times x |window|
        # candidates, so a dict lookup + arithmetic per candidate inside a
        # _score() call is the whole cost of the sweep.  beta is constant for a
        # run, so fold it in once, here.
        self.hp = {}                      # block_id -> beta * f(h)
        self.rp = {}                      # block_id -> gamma * g(r)
        self.any_r = False                # any nonzero rp in flight?

        # ONE ghost LRU, ONE budget: content -> [r, young].  Not two lists with a
        # cap each -- with a per-list cap, splitting on tau silently doubles the
        # ghost's total capacity (two lists of cap N hold 2N), so a "better
        # young/old split" is really just a bigger ghost and Delta_r stops being
        # a one-variable measurement.  This is the same discipline the design
        # already forces on C2's admission ticket.
        self.G = OrderedDict()

        self.res_sum = 0
        self.res_n = 0

    # ---- pool ----------------------------------------------------------
    def n_evictable(self):
        return len(self.hashless) + len(self.order)

    def _set_hp(self, bi):
        self.hp[bi] = self.beta * min(self.h.get(bi, 0.0), H_CAP) / H_CAP

    def _set_rp(self, bi):
        r = self.blk_r.get(bi, 0)
        if r:
            g = min(r, R_CAP) / R_CAP
            self.rp[bi] = (self.gamma_y if self.blk_gy.get(bi) else self.gamma_o) * g
        else:
            self.rp[bi] = 0.0

    def _take(self):
        """Hand out one physical block: hashless pool first, else evict."""
        if self.hashless:
            return self.hashless.popleft()
        order = self.order
        if not order:
            raise RuntimeError('pool exhausted with nothing evictable')
        w = self.window
        if w <= 1 or len(order) == 1:
            bi = next(iter(order))
            c = order.pop(bi)
        else:
            hp, rp = self.hp, self.rp
            inv = 1.0 / (min(w, len(order)) - 1)
            bs, bb = None, None
            if self.any_r:
                for rank, bi in enumerate(islice(order, w)):
                    s = rank * inv + hp[bi] - rp[bi]
                    if bs is None or s < bs:
                        bs, bb = s, bi
            else:
                for rank, bi in enumerate(islice(order, w)):
                    s = rank * inv + hp[bi]
                    if bs is None or s < bs:
                        bs, bb = s, bi
            bi = bb
            c = order.pop(bi)
        self._retire(bi, c)
        return bi

    def _retire(self, bi, content):
        """Block leaves the cache: record residency, file it in a ghost list."""
        self.C['evictions'] += 1
        if self.pol >= 2:
            res = self.seq - self.birth.get(bi, self.seq)
            self.res_sum += res
            self.res_n += 1
            self.G[content] = [self.blk_r.get(bi, 0), res < self.tau]
            self.G.move_to_end(content)
            while len(self.G) > self.ghost_cap:
                self.G.popitem(last=False)
            if len(self.G) > self.C['ghost_max']:
                self.C['ghost_max'] = len(self.G)
        if self.content_to_block.get(content) == bi:
            del self.content_to_block[content]
        self.h.pop(bi, None)
        self.blk_r.pop(bi, None)
        self.blk_gy.pop(bi, None)
        self.birth.pop(bi, None)
        self.hp.pop(bi, None)
        self.rp.pop(bi, None)

    def admit(self, content):
        """Allocate a block for a content that must now be cached."""
        bi = self._take()
        self.content_to_block[content] = bi
        self.birth[bi] = self.seq
        self.h[bi] = 0.0
        self.blk_r[bi] = 0
        self.blk_gy[bi] = False
        self.rp[bi] = 0.0
        if self.pol >= 2 and content in self.G:
            self._resurrect(bi, content)
        self._set_hp(bi)
        return bi

    def _resurrect(self, bi, content):
        r, young = self.G.pop(content)
        self.blk_r[bi] = r + 1
        self.blk_gy[bi] = young
        self.h[bi] = TICKET_H             # the C2 admission ticket
        if self.pol >= 3:
            self._set_rp(bi)
            if self.rp[bi]:
                self.any_r = True
        self.C['ghost_hit_y' if young else 'ghost_hit_o'] += 1

    def take_hashless(self):
        """A partial tail block: gets a physical block but no content, and is
        handed straight back -- so it never enters the queue or a ghost."""
        return self._take()

    # ---- hit / release -------------------------------------------------
    def touch(self, content):
        bi = self.content_to_block.get(content)
        if bi is None:
            return False
        if bi not in self.order:
            # A prefix chain has one content per position and requests are
            # driven one at a time, so this is unreachable.  Fail loudly rather
            # than silently mis-score.
            raise RuntimeError('content %r hit twice inside one residency' % (content,))
        del self.order[bi]                # in use by the request; re-enters at release
        self.C['hits'] += 1
        if self.pol >= 1:
            self.h[bi] = min(self.h.get(bi, 0.0) + 1.0, H_CAP)
            self._set_hp(bi)
        return True

    def release(self, bi, content):
        self.seq += 1
        self.order[bi] = content          # most recent -- but birth is unchanged


def _drive(q, trace_path, n, counters):
    z = np.load(trace_path)
    flat, offsets, lens = z['flat'], z['offsets'], z['lens']
    for r in range(n):
        a, b = int(offsets[r]), int(offsets[r + 1])
        blocks = flat[a:b]
        nfull = len(blocks)
        has_tail = (int(lens[r]) % BS) != 0

        k = 0
        while k < nfull and q.touch(int(blocks[k])):
            k += 1
        counters['total'] += nfull

        for j in range(k, nfull):
            q.admit(int(blocks[j]))
        tail = q.take_hashless() if has_tail else None

        # free, in the system's own order: tail first, hashed blocks reversed
        if has_tail:
            q.hashless.append(tail)
        for j in range(nfull - 1, -1, -1):
            c = int(blocks[j])
            q.release(q.content_to_block[c], c)
    return q


_PILOT = {}


def _pilot_mean_res(trace_path, max_size, window, beta, n):
    """Mean residency under the gamma=0 pilot, memoised.

    The pilot is a C2 run at the same (max_size, window, beta), so it does not
    depend on gamma_y, gamma_o or tau_frac at all -- which is the whole point
    (tau is calibrated on evidence that cannot see the gamma it parameterises).
    A C3 sweep calls run() once per (gamma_y, gamma_o, tau_frac) cell, so
    without this cache the identical pilot is recomputed once per cell: 64
    wasted full-trace runs per size in sweep_sizes.py, and it was already being
    computed a second time as the C2 reference h2.  Same numbers, half the
    wall clock."""
    key = (trace_path, max_size, window, beta, n)
    if key not in _PILOT:
        Cp = Counters()
        pq = SingleQueue(max_size, pol=2, window=window, beta=beta,
                         gamma_y=0.0, gamma_o=0.0, tau=None, counters=Cp)
        _drive(pq, trace_path, n, Cp)
        _PILOT[key] = pq.res_sum / max(1, pq.res_n)
    return _PILOT[key]


def run(trace_path, max_size, name, window=1, beta=1.0,
        gamma_y=0.0, gamma_o=0.0, tau_frac=None, ghost_cap=None,
        n_requests=None, counters=None):
    """Run one rung.  For pol>=2 and tau_frac set, tau is calibrated first on a
    gamma=0 pilot of the SAME (window, beta): with gamma=0 the ghost list a
    block lands in cannot affect any eviction, so the pilot's residency
    statistics are policy-equivalent and tau = tau_frac * mean residency does
    not depend on the very gamma it is meant to parameterise."""
    pol = {'C0': 0, 'C1': 1, 'C2': 2, 'C3': 3}[name]
    if pol == 0:
        window = 1

    z = np.load(trace_path)
    n = len(z['offsets']) - 1
    if n_requests:
        n = min(n, n_requests)

    C = counters if counters is not None else Counters()

    tau = None
    if pol >= 3 and tau_frac is not None:
        mean_res = _pilot_mean_res(trace_path, max_size, window, beta, n)
        tau = tau_frac * mean_res
        C['tau'] = tau
        C['pilot_mean_res'] = mean_res

    q = SingleQueue(max_size, pol=pol, window=window, beta=beta,
                    gamma_y=gamma_y, gamma_o=gamma_o, tau=tau,
                    ghost_cap=ghost_cap, counters=C)
    _drive(q, trace_path, n, C)
    return q


class TwoQueue:
    """2Q-lite: two LRU queues, promote on the second hit, no ghost, no p.

    This is the "structure-alone" control for the ARC decomposition.  ARC's
    +6.7pp over LRU splits into ~+1.3pp from an explicit frequency term (C1,
    window<=256) and ~+5.4pp from everything else -- the two-queue partition
    plus the B1/B2 ghost machinery.  TwoQueue keeps ONLY the partition:

      T1 (order1)  blocks seen once, LRU
      T2 (order2)  blocks seen >= 2 times, LRU, protected
      admit        -> T1
      touch in T1  -> promote to T2 (level flips at touch, placed at release)
      touch in T2  -> stay in T2, refresh MRU
      evict        -> T1's oldest first, T2 only when T1 is empty

    No ghost lists, no delta, no scoring function, no tuned partition: the T1/T2
    split is whatever the access stream grows.  If TwoQueue reaches ARC-adaptive
    (491216), the ghost + delta controller is dead weight; if it stalls at the
    C1 ceiling (462123), the partition adds nothing over a frequency term; if it
    collapses toward LRU, the naive T1 window is too short and a fixed T1 cap k
    (canonical 2Q's A1in) or the A1out ghost is load-bearing.
    """

    def __init__(self, max_size, counters=None, k=None, ghost_cap=None):
        self.max_size = max_size
        self.C = counters if counters is not None else Counters()
        self.k = k                    # T1 cap; None = natural "T1 first"
        self.ghost_cap = ghost_cap    # None = no ghost; int = A1out/Amout capacity
        self.G = OrderedDict() if ghost_cap is not None else None
        self.order1 = OrderedDict()   # T1, block_id -> content, oldest first
        self.order2 = OrderedDict()   # T2, block_id -> content, oldest first
        self.level = {}               # block_id -> 1 (T1) | 2 (T2)
        self.content_to_block = {}
        self.hashless = deque(range(1, max_size + 1))
        self.seq = 0                  # release clock

    def n_evictable(self):
        return len(self.hashless) + len(self.order1) + len(self.order2)

    def _take(self):
        if self.hashless:
            return self.hashless.popleft()
        # With a T1 cap k, shed T1's oldest while T1 is over cap (or T2 empty),
        # else shed the protected T2.  Without k, T1 first, T2 only when empty.
        if self.order1 and (self.k is None or len(self.order1) > self.k
                            or not self.order2):
            src = self.order1
            self.C['evict_t1'] += 1
        elif self.order2:
            src = self.order2
            self.C['evict_t2'] += 1
        else:
            raise RuntimeError('2Q: pool exhausted with nothing evictable')
        bi, c = src.popitem(last=False)
        self._retire(bi, c)
        return bi

    def _retire(self, bi, c):
        lv = self.level.pop(bi, None)
        if self.G is not None and lv in (1, 2):
            self.G[c] = lv
            self.G.move_to_end(c)
            while len(self.G) > self.ghost_cap:
                self.G.popitem(last=False)
        if self.content_to_block.get(c) == bi:
            del self.content_to_block[c]
        self.C['evictions'] += 1

    def admit(self, content):
        bi = self._take()
        self.content_to_block[content] = bi
        if self.G is not None:
            lv = self.G.pop(content, 0)
            if lv:
                self.level[bi] = 2    # deferred promotion via ghost (A1out/Amout)
                self.C['ghost_hit_%d' % lv] += 1
                return bi
        self.level[bi] = 1            # first access -> T1
        return bi

    def take_hashless(self):
        return self._take()

    def touch(self, content):
        bi = self.content_to_block.get(content)
        if bi is None:
            return False
        if self.level.get(bi) == 1:
            del self.order1[bi]       # second access -> promote
            self.level[bi] = 2
            self.C['promote'] += 1
        else:
            del self.order2[bi]       # repeat hit -> stay hot
        self.C['hits'] += 1
        return True

    def release(self, bi, content):
        self.seq += 1
        if self.level.get(bi) == 1:
            self.order1[bi] = content
        else:
            self.order2[bi] = content


if __name__ == '__main__':
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    TRACE = os.path.join(HERE, '..', 'dataset', 'trace_quality.npz')
    MS = 6249
    C = Counters()
    q = run(TRACE, MS, 'C0', counters=C)
    print('C0 (LRU)  hits=%d / %d = %.5f' % (C['hits'], C['total'],
                                             C['hits'] / C['total']))
    print('   real  quality/100000/LRU hit_blocks = 415879  (sweep_results.tsv)')
