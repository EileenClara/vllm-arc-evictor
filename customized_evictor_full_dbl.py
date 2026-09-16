import heapq
import os
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Tuple
from vllm.core.evictor import BlockMetaData, Evictor


class CustomizedARCEvictor(Evictor):
    CLEANUP_THRESHOLD = 50

    def __init__(self, max_size=134):
        self.max_size = max_size
        self.p = 0
        _pcf = float(os.environ.get('ARC_P_CAP_FRAC', '0.5'))
        self.p_cap = int(self.max_size * _pcf) if _pcf <= 1.0 else int(_pcf)  # C4: soft-cap p (<=1 frac, >1 absolute)
        _gcb2 = int(os.environ.get('ARC_GHOST_CAP_B2', '0'))
        self.ghost_cap_B2 = _gcb2 if _gcb2 > 0 else self.max_size
        self.T1_table = {}
        self.T2_table = {}
        self.T1_queue = []
        self.T2_queue = []
        self.B1 = OrderedDict()
        self.B2 = OrderedDict()
        self.T1_active = set()
        self.T2_active = set()
        self._pending_ghost = deque()
        # ---- T2 internal frequency counter (bypasses ghost/p, pure T2 eviction) ----
        self.t2_freq_enable = int(os.environ.get('ARC_T2_FREQ_ENABLE', '0')) == 1
        self.t2_freq_window = int(os.environ.get('ARC_T2_FREQ_WINDOW', '8'))
        self.t2_freq_decay = float(os.environ.get('ARC_T2_FREQ_DECAY', '0.5'))
        self.T2_freq = {}          # block_id -> freq (raw value at last touch, lazy-decayed)
        self.T2_last_touch = {}    # block_id -> step (global step at last touch)
        self._step = 0             # monotonic step, +1 per T2 eviction (lazy-decay clock)

    def __contains__(self, block_id):
        return block_id in self.T1_table or block_id in self.T2_table

    def evict(self, content_hash=None):
        eid, ech = -1, -1
        if content_hash is not None and content_hash in self.B1:
            d = max(1, len(self.B2) // max(1, len(self.B1)))
            self.p = min(self.p + d, self.p_cap)
            eid, ech = self._replace(content_hash)
            if eid < 0:
                raise RuntimeError("ARC: B1 hit but _replace returned -1")
            del self.B1[content_hash]
            self.T2_active.add(eid)
        elif content_hash is not None and content_hash in self.B2:
            d = max(1, len(self.B1) // max(1, len(self.B2)))
            self.p = max(self.p - d, 0)
            eid, ech = self._replace(content_hash)
            if eid < 0:
                raise RuntimeError("ARC: B2 hit but _replace returned -1")
            del self.B2[content_hash]
            self.T2_active.add(eid)
        else:
            L1 = len(self.T1_table) + len(self.T1_active) + len(self.B1)
            if L1 > self.max_size:
                # Trim B1 until L1 fits
                while (len(self.T1_table) + len(self.T1_active) +
                       len(self.B1)) > self.max_size and self.B1:
                    self.B1.popitem(last=False)
            if L1 == self.max_size or L1 > self.max_size:
                if len(self.T1_table) + len(self.T1_active) < self.max_size:
                    eid, ech = self._replace(content_hash)
                else:
                    eid, ech = self._evict_from_T1()
                    self.B1[ech] = None
                    _f = open('/tmp/ghost.log', 'a')
                    _f.write('EVICT_T1 block=' + str(eid) + ' hash=' + str(ech) + '\n')
                    _f.close()
            elif L1 < self.max_size:
                eid, ech = self._replace(content_hash)
            if eid < 0:
                raise RuntimeError(f"ARC: evict returned -1. L1={L1} "
                                   f"T1={len(self.T1_table)} T1a={len(self.T1_active)} "
                                   f"B1={len(self.B1)} T2={len(self.T2_table)}")
            if len(self._pending_ghost) > 0:
                self._pending_ghost.popleft()
                self.T2_active.add(eid)
            else:
                self.T1_active.add(eid)
        return eid, ech

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
        if block_id in self.T1_table:
            self.T1_table.pop(block_id)
            self.T2_active.add(block_id)
        elif block_id in self.T2_table:
            self.T2_table.pop(block_id)
            self.T2_active.add(block_id)
        else:
            raise ValueError(f"not tracked {block_id}")

    @property
    def num_blocks(self):
        return len(self.T1_table) + len(self.T2_table)

    def _replace(self, ch_r):
        L1 = len(self.T1_table) + len(self.T1_active)
        if self.T1_table and (
                L1 > self.p or
                (ch_r and ch_r in self.B2 and L1 == self.p)):
            bi, ch = self._evict_from_T1()
            self.B1[ch] = None
            _f = open('/tmp/ghost.log', 'a')
            _f.write('EVICT_T1 block=' + str(bi) + ' hash=' + str(ch) + '\n')
            _f.close()
            return bi, ch
        if self.T2_table:
            bi, ch = self._evict_from_T2()
            self.B2[ch] = None
            _f = open('/tmp/ghost.log', 'a')
            _f.write('EVICT_T2 block=' + str(bi) + ' hash=' + str(ch) + '\n')
            _f.close()
            return bi, ch
        if self.T1_table:
            bi, ch = self._evict_from_T1()
            self.B1[ch] = None
            _f = open('/tmp/ghost.log', 'a')
            _f.write('EVICT_T1 block=' + str(bi) + ' hash=' + str(ch) + '\n')
            _f.close()
            return bi, ch
        raise ValueError("no block to replace")

    def _evict_from_T1(self):
        while self.T1_queue:
            la, nt, bi, ch = heapq.heappop(self.T1_queue)
            if bi in self.T1_table and self.T1_table[bi].last_accessed == la:
                self.T1_table.pop(bi)
                return bi, ch
        raise ValueError("T1 empty")

    def _t2_freq_of(self, bi):
        """Lazy-decayed view of freq (read-only, no write-back)."""
        f = self.T2_freq.get(bi, 1.0)
        lt = self.T2_last_touch.get(bi, self._step)
        gap = self._step - lt
        if gap > 0:
            f = f * (self.t2_freq_decay ** gap)
        return f

    def _evict_from_T2(self):
        if not self.t2_freq_enable:
            while self.T2_queue:
                la, nt, bi, ch = heapq.heappop(self.T2_queue)
                if bi in self.T2_table and self.T2_table[bi].last_accessed == la:
                    self.T2_table.pop(bi)
                    return bi, ch
            raise ValueError("T2 empty")
        self._step += 1
        # window + min-freq eviction: pop up to K valid candidates near LRU head,
        # evict the min-freq one, push the rest back (no LRU position refresh)
        cands = []
        while self.T2_queue and len(cands) < self.t2_freq_window:
            la, nt, bi, ch = heapq.heappop(self.T2_queue)
            if bi in self.T2_table and self.T2_table[bi].last_accessed == la:
                f = self._t2_freq_of(bi)
                cands.append((f, la, nt, bi, ch))
        if not cands:
            raise ValueError("T2 empty")
        cands.sort(key=lambda x: (x[0], x[1]))  # min freq first, then LRU-oldest
        f, la, nt, bi, ch = cands[0]
        self.T2_table.pop(bi)
        self.T2_freq.pop(bi, None)
        self.T2_last_touch.pop(bi, None)
        for _, la2, nt2, bi2, ch2 in cands[1:]:
            heapq.heappush(self.T2_queue, (la2, nt2, bi2, ch2))
        return bi, ch

    def _add_to_T1(self, bi, m):
        self.T1_table[bi] = m
        heapq.heappush(self.T1_queue,
                       (m.last_accessed, -m.num_hashed_tokens, bi,
                        m.content_hash))

    def _add_to_T2(self, bi, m):
        self.T2_table[bi] = m
        heapq.heappush(self.T2_queue,
                       (m.last_accessed, -m.num_hashed_tokens, bi,
                        m.content_hash))
        if self.t2_freq_enable:
            # re-enter T2 after a hit (release): lazy-decay then +1
            # first entry into T2 (T1HIT / B1 / B2 promotion): freq = 1
            if bi in self.T2_freq:
                f = self._t2_freq_of(bi) + 1.0
            else:
                f = 1.0
            self.T2_freq[bi] = f
            self.T2_last_touch[bi] = self._step

    def _prune_ghosts(self):
        while (len(self.T1_table) + len(self.T1_active) +
               len(self.B1)) > self.max_size and self.B1:
            self.B1.popitem(last=False)
        while (len(self.T2_table) + len(self.T2_active) +
               len(self.B2)) > self.ghost_cap_B2 and self.B2:
            self.B2.popitem(last=False)

class CustomizedDBLEvictor(Evictor):
    CLEANUP_THRESHOLD = 50  # Threshold to trigger heap cleanup

    def __init__(self, max_size: int = 134, k: int = 67):
        self.max_size = max_size
        self.k = k

        # Free tables for active entries
        self.A1in_free_table: Dict[int, BlockMetaData] = {}
        self.Am_free_table: Dict[int, BlockMetaData] = {}

        # Priority queues for eviction order: (last_accessed, -tokens, block_id, content_hash)
        self.A1in_priority_queue: List[Tuple[float, int, int, int]] = []
        self.Am_priority_queue: List[Tuple[float, int, int, int]] = []

        # Ghost queues for tracking evicted entries
        self.A1out_ghost: deque[int] = deque(maxlen=self.max_size * 100)
        self.Amout_ghost: deque[int] = deque(maxlen=self.max_size * 100)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.A1in_free_table or block_id in self.Am_free_table

    def evict(self, content_hash: int = None) -> Tuple[int, int]:
        # Always evict from A1in if available, else from Am
        if len(self.A1in_free_table) > self.k:
            return self._evict_from_A1in()
        elif len(self.Am_free_table) > self.max_size - self.k:
            return self._evict_from_Am()
        elif self.A1in_free_table:
            return self._evict_from_A1in()
        if self.Am_free_table:
            return self._evict_from_Am()
        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int, 
            last_accessed: float, prev_block_id: Optional[int] = None):
        """
        Register a block for future eviction:
        - No eviction should occur here; eviction is driven externally.
        - Blocks must not already exist in A1in or Am.
        """
        # print(f"DBLCache A1: {len(self.A1in_free_table)}, Am: {len(self.Am_free_table)}")
        # Ensure this is a new registration
        assert block_id not in self.A1in_free_table and block_id not in self.Am_free_table, \
            f"Block {block_id} unexpectedly already tracked"

        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)

        # assert block_id in self.A1out_ghost or block_id in self.Amout_ghost
        # Promote from A1out ghost to Am if ghost hit
        if block_id in self.A1out_ghost:
            self.A1out_ghost.remove(block_id)
            self.Am_free_table[block_id] = meta
            heapq.heappush(
                self.Am_priority_queue,
                (last_accessed, -num_hashed_tokens, block_id, content_hash)
            )
            self._cleanup_if_necessary(self.Am_priority_queue, self.Am_free_table)
            return

        # Promote from Amout ghost to Am if ghost hit
        if block_id in self.Amout_ghost:
            self.Amout_ghost.remove(block_id)
            self.Am_free_table[block_id] = meta
            heapq.heappush(
                self.Am_priority_queue,
                (last_accessed, -num_hashed_tokens, block_id, content_hash)
            )
            self._cleanup_if_necessary(self.Am_priority_queue, self.Am_free_table)
            return

        # Insert into A1in for new block
        self.A1in_free_table[block_id] = meta
        heapq.heappush(
            self.A1in_priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash)
        )
        self._cleanup_if_necessary(self.A1in_priority_queue, self.A1in_free_table)

    def update(self, block_id: int, last_accessed: float):
        # Only update last_accessed in free tables; add will not see existing blocks
        if block_id in self.A1in_free_table:
            self.A1in_free_table[block_id].last_accessed = last_accessed
        elif block_id in self.Am_free_table:
            self.Am_free_table[block_id].last_accessed = last_accessed
        else:
            raise ValueError(f"Attempting to update non-tracked block {block_id}")

    def remove(self, block_id: int):
        if block_id in self.A1in_free_table:
            self.A1in_free_table.pop(block_id)
            self.A1out_ghost.append(block_id)
        elif block_id in self.Am_free_table:
            self.Am_free_table.pop(block_id)
            self.Amout_ghost.append(block_id)
        else:
            raise ValueError(f"Attempting to remove non-tracked block {block_id}")

    @property
    def num_blocks(self) -> int:
        return len(self.A1in_free_table) + len(self.Am_free_table)

    def _evict_from_A1in(self) -> Tuple[int, int]:
        # Evict oldest from A1in and record to ghost
        while self.A1in_priority_queue:
            last_accessed, neg_tokens, block_id, content_hash = heapq.heappop(self.A1in_priority_queue)
            if (block_id in self.A1in_free_table and
                    self.A1in_free_table[block_id].last_accessed == last_accessed):
                self.A1in_free_table.pop(block_id)
                # self.A1out_ghost.append(block_id)
                return block_id, content_hash
        raise ValueError("No usable block left in A1in to evict")

    def _evict_from_Am(self) -> Tuple[int, int]:
        # Evict oldest from Am and record to ghost
        while self.Am_priority_queue:
            last_accessed, neg_tokens, block_id, content_hash = heapq.heappop(self.Am_priority_queue)
            if (block_id in self.Am_free_table and
                    self.Am_free_table[block_id].last_accessed == last_accessed):
                self.Am_free_table.pop(block_id)
                # self.Amout_ghost.append(block_id)
                return block_id, content_hash
        raise ValueError("No usable block left in Am to evict")

    def _cleanup_if_necessary(self, queue: List[Tuple[float, int, int, int]], table: Dict[int, BlockMetaData]):
        if len(queue) > self.CLEANUP_THRESHOLD * len(table):
            self._cleanup(queue, table)

    def _cleanup(self, queue: List[Tuple[float, int, int, int]], table: Dict[int, BlockMetaData]):
        new_queue = []
        for block_id, meta in table.items():
            new_queue.append((meta.last_accessed, -meta.num_hashed_tokens, block_id, meta.content_hash))
        heapq.heapify(new_queue)

        if table is self.A1in_free_table:
            self.A1in_priority_queue = new_queue
        else:
            self.Am_priority_queue = new_queue


class DBLEvictorAdapter(CustomizedDBLEvictor):
    def __init__(self, max_size=134, k=None):
        if k is None:
            k = int(max_size * 0.5)
        super().__init__(max_size=max_size, k=k)

    """ARC-compatible adapter so DBL can run behind ARCBlockQueue.

    Maps ARC naming onto DBL internals; ghost promotion (A1out/Amout)
    is handled by CustomizedDBLEvictor.add(). B1/B2/p are inert (DBL has
    no content-hash ghosts and no adaptive parameter).
    """

    @property
    def T1_table(self):
        return self.A1in_free_table

    @property
    def T2_table(self):
        return self.Am_free_table

    @property
    def B1(self):
        return {}

    @property
    def B2(self):
        return {}

    @property
    def T1_active(self):
        return set()

    @property
    def T2_active(self):
        return set()

    @property
    def p(self):
        return 0

    @property
    def T1_queue(self):
        return self.A1in_priority_queue

    @property
    def T2_queue(self):
        return self.Am_priority_queue

    def _add_to_T1(self, bi, m):
        # goes through add() so A1out/Amout ghost promotion still fires
        self.add(bi, m.content_hash, m.num_hashed_tokens, m.last_accessed)

    def _add_to_T2(self, bi, m):
        self._add_to_T1(bi, m)

    def _prune_ghosts(self):
        pass
