import heapq
import os
from collections import OrderedDict, deque
from vllm.core.evictor import BlockMetaData, Evictor


class CustomizedARCEvictor(Evictor):
    CLEANUP_THRESHOLD = 50

    def __init__(self, max_size=134):
        self.max_size = max_size
        self.p = 0
        _pcf = float(os.environ.get('ARC_P_CAP_FRAC', '0.5'))
        self.p_cap = int(self.max_size * _pcf) if _pcf <= 1.0 else int(_pcf)  # C4: soft-cap p (<=1 frac, >1 absolute)
        self.T1_table = {}
        self.T2_table = {}
        self.T1_queue = []
        self.T2_queue = []
        self.B1 = OrderedDict()
        self.B2 = OrderedDict()
        self.T1_active = set()
        self.T2_active = set()
        self._pending_ghost = deque()

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

    def _evict_from_T2(self):
        while self.T2_queue:
            la, nt, bi, ch = heapq.heappop(self.T2_queue)
            if bi in self.T2_table and self.T2_table[bi].last_accessed == la:
                self.T2_table.pop(bi)
                return bi, ch
        raise ValueError("T2 empty")

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

    def _prune_ghosts(self):
        while (len(self.T1_table) + len(self.T1_active) +
               len(self.B1)) > self.max_size and self.B1:
            self.B1.popitem(last=False)
        while (len(self.T2_table) + len(self.T2_active) +
               len(self.B2)) > self.max_size and self.B2:
            self.B2.popitem(last=False)
