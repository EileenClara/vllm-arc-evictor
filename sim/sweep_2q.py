"""2Q vs ARC -- does the two-queue structure alone carry ARC's edge?

The control this investigation still lacks.  ARC's +6.7pp over LRU decomposes
into ~+1.3pp from an explicit frequency term (C1 at window<=256) and ~+5.4pp
from the partition + ghost machinery.  DBL(best k) already isolates "partition
held fixed, ghost present but not driving it".  TwoQueue isolates "partition
present, ghost absent" -- promote on the live second hit, no B1/B2, no delta.

Anchors (block-level hits out of 1,124,980):
  C0 LRU                415,885
  C1 ceiling (K=2048)   462,123   <- single-queue frequency term, no ghost
  DBL best (k=2500)    ~480,564   <- partition + ghost, fixed split
  ARC-adaptive          491,216
  ARC-p* (frozen)       500,311   <- the honest ceiling
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
C1_CEIL = 462123
DBL_BEST = 480564   # k=2500, from sweep_dbl (untuned 470507 + 0.894pp)
ARC = 491216
ARCPS = 500311

# Same grid DBL was swept on, so 2Q(k) is directly comparable to DBL(k).
KS = [1, 156, 312, 625, 1250, 1874, 2500, 3124, 3749, 4374, 5000, 5624, 6248]


def run_2q(k, n):
    C = Counters()
    q = TwoQueue(MS, counters=C, k=k)
    _drive(q, TRACE, n, C)
    return C, q


def main():
    z = np.load(TRACE)
    n = len(z['offsets']) - 1
    t0 = time.time()

    # natural (no cap) for reference -- known to collapse below LRU
    Cn, qn = run_2q(None, n)
    print('2Q natural (no T1 cap): hits=%d  (%.5f)  vs LRU %+.3f pp'
          % (Cn['hits'], Cn['hits'] / Cn['total'],
             100.0 * (Cn['hits'] - C0) / TOTAL))
    print('    evict_t1=%d evict_t2=%d promote=%d'
          % (Cn['evict_t1'], Cn['evict_t2'], Cn['promote']))
    print()

    print('k sweep (T1 cap = k):')
    print('  %-7s %-10s %9s %10s %10s' % ('k', 'k/max_size', 'hits', 'vs LRU', 'vs ARC'))
    rows = []
    for k in KS:
        C, q = run_2q(k, n)
        rows.append((k, C['hits']))
        print('  %-7d %-10.3f %9d %+9.3f%% %+9.3f%%'
              % (k, k / MS, C['hits'],
                 100.0 * (C['hits'] - C0) / TOTAL,
                 100.0 * (C['hits'] - ARC) / TOTAL), flush=True)

    best_k, best_h = max(rows, key=lambda r: r[1])
    print()
    print('=' * 78)
    print('2Q best k = %d (%.1f%% of the pool) -> %d hits (%.5f)'
          % (best_k, 100.0 * best_k / MS, best_h, best_h / TOTAL))
    print()
    for lbl, h in [('C0 LRU', C0), ('C1 ceiling', C1_CEIL), ('DBL best', DBL_BEST),
                   ('ARC-adaptive', ARC), ('ARC-p*', ARCPS)]:
        print('  %-13s %8d   %+.4f pp vs LRU' % (lbl, h, 100.0 * (h - C0) / TOTAL))
    print('  %-13s %8d   %+.4f pp vs LRU' % ('2Q best', best_h,
                                             100.0 * (best_h - C0) / TOTAL))
    print('  vs ARC-adaptive %+d   vs ARC-p* %+d   vs DBL best %+d'
          % (best_h - ARC, best_h - ARCPS, best_h - DBL_BEST))
    print('\nelapsed %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
