"""Decisive test: is the ARC vs 2Q gap at p=0 caused by B2 ghost starvation?

ARC's _prune_ghosts bounds the frequency ghost by
    |T2_table| + |T2_active| + |B2| <= ghost_cap_B2   (default ghost_cap_B2 = max_size)
so at p=0, where T2 fills the cache, B2 is pruned to ~0 and ARC loses exactly
the frequency-ghost memory that 2Q's single ghost keeps.

If that coupling is the whole story, lifting ghost_cap_B2 to 2*max_size (so B2
can hold ~max_size even when T2 is full) should lift ARC-p*(p=0) from 500087
up toward 2Q+ghost(k=0) = 505087.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import arcsim
from arcsim import COUNTS, run

TRACE = os.path.join(HERE, '..', 'dataset', 'trace_quality.npz')
MS = 6249
TOTAL = 1124980
TWOQ_K0 = 505087


def one(p, gcap):
    # set before the evictor is constructed inside run()
    if gcap is None:
        os.environ.pop('ARC_GHOST_CAP_B2', None)
    else:
        os.environ['ARC_GHOST_CAP_B2'] = str(gcap)
    COUNTS.clear()
    shim = run(TRACE, max_size=MS, verbose=False, p_freeze=p)
    return {
        'p': p,
        'gcap': gcap,
        'hits': COUNTS['REMOVE'],
        'B1_hit': COUNTS['GHOST_B1'],
        'B2_hit': COUNTS['GHOST_B2'],
        'B1_final': len(shim.evictor.B1),
        'B2_final': len(shim.evictor.B2),
        'T2_final': len(shim.evictor.T2_table) + len(shim.evictor.T2_active),
        'T1_final': len(shim.evictor.T1_table) + len(shim.evictor.T1_active),
    }


t0 = time.time()
print('%-4s %-10s %-9s %-8s %-8s %-8s %-8s %-8s %-8s %-10s'
      % ('p', 'ghost_cap', 'hits', 'B1_hit', 'B2_hit', 'B1_end', 'B2_end',
         'T1_end', 'T2_end', 'vs 2Q k=0'))
for gcap in [None, 2 * MS, 3 * MS]:
    for p in [0, 300]:
        r = one(p, gcap)
        print('%-4d %-10s %-9d %-8d %-8d %-8d %-8d %-8d %-8d %+10d'
              % (r['p'], str(r['gcap']), r['hits'], r['B1_hit'], r['B2_hit'],
                 r['B1_final'], r['B2_final'], r['T1_final'], r['T2_final'],
                 r['hits'] - TWOQ_K0), flush=True)

print('\nelapsed %.1fs' % (time.time() - t0))
