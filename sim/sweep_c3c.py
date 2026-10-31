"""gamma_y scanned at EVERY tau, not only at the tie-winner.

sweep_c3b.py picked its stage-2 operating point with max(grid, key=hits), and
in stage 1 the gamma_o=0 column is 429759 at all four tau -- a four-way tie.
max() returns the FIRST maximal element, so stage 2 would have run at
tau_frac=0.5 (tau=5014).  That is below the residency distribution's p10
(6624), so the young list is all but empty there and gamma_y has nothing to
act on: stage 2 would have printed "gamma_y INERT" for the third time, from the
same cause as bug 2 and by a route no value could reveal -- only the tie-break
order.

So instead of one operating point, this scans the full tau x gamma_y grid at
gamma_o=0 (gamma_o is uniformly harmful in stage 1, monotonically in every tau,
so the old arm is settled and does not need re-opening).

Every cell also reports gy_hits, the number of evictions actually filed young.
A cell with gy_hits ~ 0 cannot say anything about gamma_y no matter what its
hit count is, and printing that column is what makes the difference between
"gamma_y does not help" and "gamma_y was never switched on" visible in the
output rather than in my memory of what the grid was supposed to cover.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from policies import Counters, run

TRACE = os.path.join(HERE, '..', 'dataset', 'trace_quality.npz')
MS = 6249
K, B = 256, 8.0
C0_HITS, C1_HITS, C2_HITS = 415885, 430710, 429759
TOTAL = 1124980
FRACS = (0.5, 0.75, 1.0, 1.5)
GYS = (0.0, 0.25, 0.5, 1.0, 2.0)


def main():
    t0 = time.time()
    print('=' * 96)
    print('C3: full tau x gamma_y grid at gamma_o = 0     '
          '(C0=%d C1=%d C2=%d)' % (C0_HITS, C1_HITS, C2_HITS))
    print('=' * 96)
    print()
    print('%-9s %-9s' % ('tau_frac', 'tau'), end='')
    for gy in GYS:
        print('%14s' % ('gy=%.2f' % gy), end='')
    print('%12s' % 'gy_hits')
    print()

    grid = []
    for frac in FRACS:
        cells = []
        for gy in GYS:
            C = Counters()
            run(TRACE, MS, 'C3', counters=C, window=K, beta=B,
                gamma_y=gy, gamma_o=0.0, tau_frac=frac)
            # gy_hits belongs to THIS cell, not to the row: a run with a
            # different gamma_y evicts different blocks, so the number filed
            # young changes with gamma_y too.  Carrying one row-level value
            # would misreport reachability on exactly the cells that matter.
            grid.append(dict(frac=frac, gy=gy, hits=C['hits'], tau=C['tau'],
                             gyh=C['ghost_hit_y'], goh=C['ghost_hit_o']))
            cells.append(C['hits'])
        print('%-9.2f %-9.1f' % (frac, grid[-1]['tau']), end='')
        for h in cells:
            print('%14d' % h, end='')
        print(flush=True)

    print()
    print('  --- reachability: a cell with gy_hits ~ 0 cannot test gamma_y ---')
    print('  %-9s %-9s %-11s %-11s %s'
          % ('tau_frac', 'tau', 'gy_hits', 'go_hits', 'verdict'))
    reach_fracs = []
    for frac in FRACS:
        # reachability is a property of tau, so read it off the gy=0 cell of the
        # row -- that cell always exists and is the one tau was calibrated for.
        r = [g for g in grid if g['frac'] == frac and g['gy'] == 0.0][0]
        ok = r['gyh'] > 1000
        if ok:
            reach_fracs.append(frac)
        print('  %-9.2f %-9.1f %-11d %-11d %s'
              % (frac, r['tau'], r['gyh'], r['goh'],
                 'young arm reachable' if ok else 'YOUNG ARM EMPTY'))

    print()
    best = max(grid, key=lambda g: g['hits'])
    reach = [g for g in grid if g['frac'] in reach_fracs]
    print('  best cell overall: tau_frac=%.2f gamma_y=%.2f hits=%d'
          % (best['frac'], best['gy'], best['hits']))
    if reach:
        br = max(reach, key=lambda g: g['hits'])
        col = sorted(g['hits'] for g in reach if g['frac'] == br['frac'])
        print('  best reachable:    tau_frac=%.2f gamma_y=%.2f hits=%d'
              % (br['frac'], br['gy'], br['hits']))
        print('  at tau_frac=%.2f the full gamma_y column (incl. gy=0) is %s -> %s'
              % (br['frac'], col,
                 'ACTIVE (gamma_y changes the outcome)' if len(set(col)) > 1
                 else 'INERT (gamma_y changes nothing)'))
    else:
        print('  NO tau made the young arm reachable -- gamma_y is untested,')
        print('  which is a grid failure, NOT a negative result.')

    h3 = best['hits']
    print()
    print('  C3 best = %d (%.5f)' % (h3, h3 / TOTAL))
    print('  Delta_r = C3 - C2 = %+d blocks = %+.4f pp'
          % (h3 - C2_HITS, 100.0 * (h3 - C2_HITS) / TOTAL))
    print('  pre-registered |Delta_r| < 0.3pp -> %s'
          % ('FIRED' if abs(100.0 * (h3 - C2_HITS) / TOTAL) < 0.3
             else 'not fired'))
    print('\nelapsed %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
