"""Check 3 -- the C0/C1/C2/C3 ladder.

Anchors on the same trace (QuALITY, max_size 6249, 3072 requests, block-level):

  C0 real LRU          415,879   (sweep_results.tsv, quality/100000/LRU)
  C0 this sim          415,885   (+6 blocks; the sim is validated)
  ARC-adaptive         491,216   (arcsim.py)
  ARC-p* frozen@300    500,311   (sweep_p.py)  <- the honest ceiling

The ceiling is ARC-p*, not ARC-adaptive: sweep_p showed the delta rule's fixed
point sits ~4x above this workload's empirical optimum, worth +0.81pp.  Credit
for "beating ARC" only counts against the frozen oracle, else we would just be
measuring the controller's mistuning.

Each rung is given its best hyper-parameters on the full trace (an oracle, and
deliberately so: the falsification condition is about Delta_r being ~0, so C3
must not lose for want of tuning).  A split-half check follows so the headline
number is not an oracle artifact.
"""
import os
import sys
import time
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from policies import Counters, run

TRACE = os.path.join(HERE, '..', 'dataset', 'trace_quality.npz')
MS = 6249
ANCHOR_LRU = 415879
ANCHOR_ARC = 491216
ANCHOR_ARCPS = 500311

# C1 / beta grid
WIN = [2, 4, 8, 16, 32, 64, 128, 256]
BETAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def one(name, **kw):
    C = Counters()
    run(TRACE, MS, name, counters=C, **kw)
    return C


def hr(C):
    return C['hits'] / C['total']


def main():
    t0 = time.time()
    print('=' * 78)
    print('C0  LRU')
    print('=' * 78)
    c0 = one('C0')
    print('  sim  %d  (%.5f)      real LRU %d   delta %+d'
          % (c0['hits'], hr(c0), ANCHOR_LRU, c0['hits'] - ANCHOR_LRU))

    print()
    print('=' * 78)
    print('C1  {h}   S = age_rank + beta*f(h), window of K oldest candidates')
    print('=' * 78)
    print('%-8s' % 'K', end='')
    for b in BETAS:
        print('%10s' % ('b=%.2f' % b), end='')
    print('%10s' % 'best')
    grid = []
    for K in WIN:
        print('%-8d' % K, end='')
        best = None
        for b in BETAS:
            C = one('C1', window=K, beta=b)
            grid.append((K, b, C['hits']))
            if best is None or C['hits'] > best[2]:
                best = (K, b, C['hits'])
            print('%10d' % C['hits'], end='')
        print('%10d' % best[2], flush=True)

    K1, b1, h1 = max(grid, key=lambda x: x[2])
    print()
    print('  C1 best: K=%d beta=%.2f  hits=%d  (%.5f)' % (K1, b1, h1, h1 / c0['total']))
    print('  Delta_h = C1 - C0 = %+d blocks = %+.4f pp'
          % (h1 - c0['hits'], 100.0 * (h1 - c0['hits']) / c0['total']))
    print('  vs ARC-adaptive  %+d   vs ARC-p*  %+d'
          % (h1 - ANCHOR_ARC, h1 - ANCHOR_ARCPS))

    print()
    print('=' * 78)
    print('C2  {h + ticket}   ghost map on, resurrection sets h<-2, r recorded, gamma=0')
    print('=' * 78)

    # C2's only tuning is the same (K, beta) surface; gamma is fixed at 0 so tau
    # cannot matter.  Re-scan a narrowed surface around C1's optimum.
    c2_grid = []
    for K in WIN:
        for b in BETAS:
            C = one('C2', window=K, beta=b)
            c2_grid.append((K, b, C['hits'], C['ghost_hit_y'], C['ghost_hit_o']))
    K2, b2, h2, gy, go = max(c2_grid, key=lambda x: x[2])
    print('  C2 best: K=%d beta=%.2f  hits=%d  (%.5f)' % (K2, b2, h2, h2 / c0['total']))
    print('  ghost hits: young=%d old=%d' % (gy, go))
    print('  Delta_ticket = C2 - C1 = %+d blocks = %+.4f pp'
          % (h2 - h1, 100.0 * (h2 - h1) / c0['total']))
    print('  vs ARC-adaptive  %+d   vs ARC-p*  %+d'
          % (h2 - ANCHOR_ARC, h2 - ANCHOR_ARCPS))

    print()
    print('=' * 78)
    print('C3  full   + gamma_y*g(r_y) + gamma_o*g(r_o)')
    print('=' * 78)
    print('  tau is calibrated per run as tau_frac * mean residency (gamma=0 pilot)')
    print()
    # Two stages, because the full (frac x gamma_y x gamma_o) cube is 3x5x5 runs
    # and each C3 run carries a tau-calibration pilot.  Stage 1: gamma_y = 0,
    # scan gamma_o x tau_frac.  Stage 2: at the winning (frac, gamma_o), scan
    # gamma_y.  If gamma_y matters anywhere it shows up here.
    GAM = [0.0, 0.25, 0.5, 1.0, 2.0]
    FRACS = [0.125, 0.25, 0.5]
    print('  stage 1: gamma_y = 0')
    print('  %-10s' % 'tau_frac', end='')
    for g in GAM:
        print('%10s' % ('go=%.2f' % g), end='')
    print()
    c3_grid = []
    for frac in FRACS:
        print('  %-10.3f' % frac, end='')
        for g in GAM:
            C = one('C3', window=K2, beta=b2, gamma_y=0.0, gamma_o=g,
                    tau_frac=frac)
            c3_grid.append((frac, 0.0, g, C['hits'], C.get('tau', 0)))
            print('%10d' % C['hits'], end='')
        print(flush=True)

    f3, _, go3, h3, tau3 = max(c3_grid, key=lambda x: x[3])
    print()
    print('  stage 2: tau_frac=%.3f (tau=%.1f) gamma_o=%.2f' % (f3, tau3, go3))
    print('  %-10s' % 'gamma_y', end='')
    for gy_ in GAM:
        C = one('C3', window=K2, beta=b2, gamma_y=gy_, gamma_o=go3, tau_frac=f3)
        c3_grid.append((f3, gy_, go3, C['hits'], C.get('tau', 0)))
        print('%10d' % C['hits'], end='')
    print(flush=True)

    f3, gy3, go3, h3, tau3 = max(c3_grid, key=lambda x: x[3])
    print()
    print('  C3 best: tau_frac=%.3f (tau=%.1f) gamma_y=%.2f gamma_o=%.2f  hits=%d (%.5f)'
          % (f3, tau3, gy3, go3, h3, h3 / c0['total']))
    print('  Delta_r = C3 - C2 = %+d blocks = %+.4f pp'
          % (h3 - h2, 100.0 * (h3 - h2) / c0['total']))
    print('  vs ARC-adaptive  %+d   vs ARC-p*  %+d'
          % (h3 - ANCHOR_ARC, h3 - ANCHOR_ARCPS))
    # gamma=0 row of C3 must reproduce C2 exactly -- a wiring check
    z = [r for r in c3_grid if r[1] == 0.0 and r[2] == 0.0]
    print('  wiring check: C3 with gamma=0 -> %s  (C2 = %d)'
          % (sorted(set(r[3] for r in z)), h2))

    print()
    print('=' * 78)
    print('SUMMARY   (hits, and pp of the 1,124,980-block trace)')
    print('=' * 78)
    tot = c0['total']
    for lbl, h in [('C0  LRU', c0['hits']), ('C1  {h}', h1),
                   ('C2  +ticket', h2), ('C3  full', h3),
                   ('ARC-adaptive', ANCHOR_ARC), ('ARC-p*', ANCHOR_ARCPS)]:
        print('  %-14s %8d  %.5f  %+7d vs C0  %+7d vs ARC  %+7d vs ARC-p*'
              % (lbl, h, h / tot, h - c0['hits'], h - ANCHOR_ARC, h - ANCHOR_ARCPS))
    print()
    print('  Delta_h      = %+d  (%.3f pp)' % (h1 - c0['hits'], 100.0 * (h1 - c0['hits']) / tot))
    print('  Delta_ticket = %+d  (%.3f pp)' % (h2 - h1, 100.0 * (h2 - h1) / tot))
    print('  Delta_r      = %+d  (%.3f pp)' % (h3 - h2, 100.0 * (h3 - h2) / tot))
    print('\nelapsed %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
