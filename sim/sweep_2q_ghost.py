"""2Q-lite + A1out/Amout ghost (deferred promotion, no delta) vs ARC-p*.

Does ghost-as-memory close the 1.75pp gap from 2Q-lite to the frozen-p ceiling?
TwoQueue with ghost_cap records evicted blocks (T1 -> G[1], T2 -> G[2]) and
re-admits a ghost hit straight to T2 -- ARC's B1/B2 admission ticket, with the
delta controller left OUT.  If this reaches ARC-p* (500311) at a small k, the
decomposition closes: partition + ghost-memory = the whole 7.50pp, delta = 0.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from policies import Counters, TwoQueue, _drive

TRACE = os.path.join(HERE, '..', 'dataset', 'trace_quality.npz')
MS = 6249
TOTAL = 1124980
C0 = 415885
ARC = 491216
ARCPS = 500311
TWOQ_BEST = 482837   # 2Q-lite best k=1874, no ghost

KS = [1, 156, 312, 625, 1250, 1874, 2500, 3124, 3749, 4374, 5000, 5624, 6248]
GHOST_CAP = MS       # B1+B2 <= cache size, matching ARC's bound


def run_2q(k, ghost_cap, n):
    C = Counters()
    q = TwoQueue(MS, counters=C, k=k, ghost_cap=ghost_cap)
    _drive(q, TRACE, n, C)
    return C, q


def main():
    z = np.load(TRACE)
    n = len(z['offsets']) - 1
    t0 = time.time()
    print('2Q-lite + ghost (A1out/Amout, cap=%d), k sweep:' % GHOST_CAP)
    print('  %-7s %-10s %9s %10s %10s %10s'
          % ('k', 'k/max', 'hits', 'vs LRU', 'vs ARC-p*', 'ghost_hits'))
    rows = []
    for k in KS:
        C, q = run_2q(k, GHOST_CAP, n)
        gh = C['ghost_hit_1'] + C['ghost_hit_2']
        rows.append((k, C['hits']))
        print('  %-7d %-10.3f %9d %+9.3f%% %+9.3f%% %9d'
              % (k, k / MS, C['hits'],
                 100.0 * (C['hits'] - C0) / TOTAL,
                 100.0 * (C['hits'] - ARCPS) / TOTAL, gh), flush=True)

    best_k, best_h = max(rows, key=lambda r: r[1])
    print()
    print('=' * 78)
    print('2Q+ghost best k=%d -> %d (%.5f)' % (best_k, best_h, best_h / TOTAL))
    print('  vs 2Q-lite best (482837, no ghost)  %+d  %+.3f pp'
          % (best_h - TWOQ_BEST, 100.0 * (best_h - TWOQ_BEST) / TOTAL))
    print('  vs ARC-adaptive (491216)            %+d  %+.3f pp'
          % (best_h - ARC, 100.0 * (best_h - ARC) / TOTAL))
    print('  vs ARC-p* (500311)                  %+d  %+.3f pp'
          % (best_h - ARCPS, 100.0 * (best_h - ARCPS) / TOTAL))

    # the matched small-T1 point: k=312 vs ARC-p* p=300
    Ck, _ = run_2q(312, GHOST_CAP, n)
    print('  k=312 with ghost -> %d   (2Q-lite no-ghost k=312 = 416252,'
          ' ARC-p* p=300 = 500311)' % Ck['hits'])
    print('\nelapsed %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
