# -*- coding: utf-8 -*-
"""Customized2QGhostEvictor — 2Q-lite + single shared ghost (2Q+ghost).

The offline-validated 2Q+ghost policy (trace_rebuild/sim/policies.py TwoQueue
with k=0 and ghost_cap=max_size) expressed as a vLLM V1 Evictor.  It is a
drop-in replacement for CustomizedARCEvictor behind ARCBlockQueue: the shell
needs only a new VLLM_EVICTOR dispatch branch, nothing else.

Structure (same attribute surface as CustomizedARCEvictor, so ARCBlockQueue's
direct manipulation keeps working):

  T1_table / T1_active   blocks seen once        (freeable / in-use)
  T2_table / T2_active   blocks seen >= 2 times  (freeable / in-use)
  B1                     THE single shared ghost (content_hash -> level)
  B2                     always empty (keeps check_ghost_hits' B2 branch inert)
  p / p_cap              always 0 (no adaptive partition)

Policy (offline TwoQueue, k=0):
  evict   -> T1 oldest first, T2 only when T1 is empty
  ghost   -> evicted content files into B1 (level 1 from T1, 2 from T2),
             capped at max_size; a ghost hit re-admits straight to T2
  hit     -> a T1 hit promotes to T2 (shell remove() + _promote_on_release)

The ghost CONSULT happens in ARCBlockQueue.check_ghost_hits (which reads
ev.B1), not here: this evictor only WRITES B1 on eviction.  The two stay
consistent because B1 keys are hash(pure BlockHash), exactly what
check_ghost_hits hashes (the group-id-stripping fix in ARCBlockQueue._hash).
"""

import heapq
import os
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Tuple

from vllm.core.evictor import BlockMetaData, Evictor

# Opt-in eviction trace, off by default.  Set VLLM_GHOST_LOG=<path> to record
# which queue each eviction came from.  The trace is write-only and never feeds
# back into an eviction decision.
GHOST_LOG = os.environ.get('VLLM_GHOST_LOG', '')


def _trace(level, block_id, content_hash):
    if GHOST_LOG:
        with open(GHOST_LOG, 'a') as f:
            f.write(f'EVICT_T{level} block={block_id} hash={content_hash}\n')


class Customized2QGhostEvictor(Evictor):
    CLEANUP_THRESHOLD = 50

    def __init__(self, max_size=134, k=0):
        self.max_size = max_size
        self.k = k                      # T1 cap; k=0 means always evict T1 first
        self.p = 0                      # inert; assigned by check_ghost_hits
        self.p_cap = 0                  # inert; keeps p pinned at 0
        self.T1_table: Dict[int, BlockMetaData] = {}
        self.T2_table: Dict[int, BlockMetaData] = {}
        self.T1_queue: List[Tuple[float, int, int, int]] = []
        self.T2_queue: List[Tuple[float, int, int, int]] = []
        self.B1: OrderedDict = OrderedDict()    # single shared ghost (content_hash -> level)
        self.B2: OrderedDict = OrderedDict()    # empty, inert
        self.T1_active: set = set()
        self.T2_active: set = set()

    def __contains__(self, block_id):
        return block_id in self.T1_table or block_id in self.T2_table

    def evict(self, content_hash=None):
        # 2Q-lite k=0: shed T1's oldest first, T2 only when T1 is empty.
        if self.T1_table:
            bi, ch = self._evict_from_T1()
            self.B1[ch] = 1
            _trace(1, bi, ch)
        elif self.T2_table:
            bi, ch = self._evict_from_T2()
            self.B1[ch] = 2
            _trace(2, bi, ch)
        else:
            raise ValueError("2Q+ghost: no usable block to evict")
        self._prune_ghosts()
        return bi, ch

    def add(self, block_id, content_hash, num_hashed_tokens, last_accessed,
            prev_block_id=None):
        m = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)
        if block_id in self.T1_active:
            self.T1_active.remove(block_id)
            self._add_to_T1(block_id, m)
        elif block_id in self.T2_active:
            self.T2_active.remove(block_id)
            self._add_to_T2(block_id, m)
        else:
            self._add_to_T1(block_id, m)
            self._prune_ghosts()

    def update(self, block_id, last_accessed):
        pass

    def remove(self, block_id):
        # hit on a freeable block: T1 -> T2 promotion, T2 -> stays T2.
        if block_id in self.T1_table:
            self.T1_table.pop(block_id)
            self.T2_active.add(block_id)
        elif block_id in self.T2_table:
            self.T2_table.pop(block_id)
            self.T2_active.add(block_id)
        else:
            raise ValueError(f"2Q+ghost: block {block_id} not tracked")

    @property
    def num_blocks(self):
        return len(self.T1_table) + len(self.T2_table)

    def _evict_from_T1(self):
        while self.T1_queue:
            la, nt, bi, ch = heapq.heappop(self.T1_queue)
            if bi in self.T1_table and self.T1_table[bi].last_accessed == la:
                self.T1_table.pop(bi)
                return bi, ch
        raise ValueError("2Q+ghost: T1 empty")

    def _evict_from_T2(self):
        while self.T2_queue:
            la, nt, bi, ch = heapq.heappop(self.T2_queue)
            if bi in self.T2_table and self.T2_table[bi].last_accessed == la:
                self.T2_table.pop(bi)
                return bi, ch
        raise ValueError("2Q+ghost: T2 empty")

    def _add_to_T1(self, bi, m):
        self.T1_table[bi] = m
        heapq.heappush(self.T1_queue,
                       (m.last_accessed, -m.num_hashed_tokens, bi, m.content_hash))

    def _add_to_T2(self, bi, m):
        self.T2_table[bi] = m
        heapq.heappush(self.T2_queue,
                       (m.last_accessed, -m.num_hashed_tokens, bi, m.content_hash))

    def _prune_ghosts(self):
        # single shared ghost, capped at max_size (offline ghost_cap = max_size)
        while len(self.B1) > self.max_size:
            self.B1.popitem(last=False)
