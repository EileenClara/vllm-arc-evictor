"""
ARCBlockQueue —— ARC 驱逐策略在 vLLM V1 上的适配层（L3）

作用：
  V1 引擎的 block_pool 固定调用 FreeKVCacheBlockQueue 的接口
  （popleft_n / remove / prepend_n / append_n / get_all_free_blocks）。
  我们无法像 V0 那样"换一个 Evictor 子类"（V1 驱逐逻辑与双向链表数据结构
  耦合在 FreeKVCacheBlockQueue 里），所以写一个同名接口的"壳"，内部把
  驱逐决策委托给论文源码的 CustomizedARCEvictor（L2）。

结构：
  - hashed block（有内容哈希、可被前缀缓存复用）  → 交给 evictor（T1/T2/B1/B2/p）
  - hashless block（未写满 16 token，无身份）     → 独立 _hashless_pool（FIFO）

对应 V0 的设计：V0 里无 hash 块由独立的 _hashless_allocator 管理，不进
ARC 的账本；这里用 _hashless_pool 还原了这一语义。
"""

import os
import time
from collections import deque
from vllm.core.customized_evictor import CustomizedARCEvictor
from vllm.v1.core.kv_cache_utils import KVCacheBlock, get_block_hash

# 通过环境变量 VLLM_EVICTOR 选择算法内核，实现 ARC/LRU/DBL 三方公平对比。
# 同一个"壳"，只有 _Evictor 不同——保证对比时唯一变量是驱逐算法。
if os.environ.get('VLLM_EVICTOR', 'ARC') == 'LRU':
    from vllm.core.lru_evictor import LRUEvictor as _Evictor
elif os.environ.get('VLLM_EVICTOR', 'ARC') == 'DBL':
    from vllm.core.customized_evictor import DBLEvictorAdapter as _Evictor
else:
    _Evictor = CustomizedARCEvictor


class ARCBlockQueue:
    """Adaptive Replacement Cache queue for V1.

    T1_table/T2_table must ONLY contain free blocks (ref_cnt == 0).
    Two pools, mirroring V0's design:
      - blocks WITH hash    -> evictor (T1/T2/B1/B2/p bookkeeping)
      - blocks WITHOUT hash -> self._hashless_pool (independent FIFO,
        never enters T1/T2, never ghost-tracked, never evicted into B1/B2)
    """

    def __init__(self, blocks):
        # 引擎传入的全部 KV cache block（V1 只创建 num_gpu_blocks 个）
        self._blocks = blocks
        # 算法内核：ARC / LRU / DBL 之一，负责 T1/T2/B1/B2/p 或等价的驱逐决策
        self.evictor = _Evictor(max_size=len(blocks))
        # 幽灵命中后的"待路由 T2"集合：check_ghost_hits 命中 B1/B2 时加入 hash，
        # 等该 block 真正写入 hash（on_block_hashed）时消费。
        self._pending_t2 = set()
        # 记录每个 pending hash 的来源（B1 还是 B2），用于日志拆解
        self._pending_t2_src = {}          # hash -> 'B1'|'B2' (ghost hit origin)
        # 记录"释放时应该进 T2"的 block：touch 命中（remove）或幽灵命中（on_block_hashed）
        # 时写入，append_n 释放时消费。来源标记便于拆解 ADD_T2 的构成。
        self._promote_on_release = {}      # block_id -> 'B1'|'B2'|'T1HIT'|'T2HIT'
        # 无 hash 块独立池（FIFO），永不被幽灵追踪、永不被驱逐进 B1/B2
        self._hashless_pool: deque[int] = deque()
        # 单调自增的"释放序号"，作为 last_accessed 的替代：
        # 原版链表按释放顺序排队，而 time.monotonic() 在同一批释放里会相同
        # （导致 heap 退化成按 block_id 排序，丢失真实 LRU 顺序）。
        self._release_seq = 0              # monotonic release order (LRU fidelity)
        # 审计采样计数器：验证"检查发生时 hashless 池是否为空"（验证 1 用）
        self._check_count = 0              # sampling counter for hashless-pool audit
        # 当前空闲块数（= hashless 池 + evictor 内 T1/T2 的 free 块数）
        self.num_free_blocks = len(blocks)
        now = time.monotonic()
        # 初始化：把引擎给的块按有无 hash 分流。
        # 新建的块通常都无 hash → 全进 hashless 池；有 hash 的（极少）进 evictor。
        for b in blocks:
            if b.block_hash is None:
                self._hashless_pool.append(b.block_id)
            else:
                self.evictor.add(b.block_id, self._hash(b),
                                 b.block_hash_num_tokens or 0, now)

    @staticmethod
    def _hash(b):
        """统一哈希表示。

        V1 里 b.block_hash 是 BlockHashWithGroupId = 纯 block_hash + 4 字节分组 id。
        而 check_ghost_hits 从 request.block_hashes 拿到的是纯 BlockHash。
        两者 hash() 结果不同会永远对不上（幽灵命中失效的根因之一）。
        这里用 get_block_hash() 去掉分组 id，保证两侧可比。
        """
        if b.block_hash is None:
            return 0
        # b.block_hash is BlockHashWithGroupId (pure block_hash + 4-byte group
        # id). check_ghost_hits uses hash(pure BlockHash) from
        # request.block_hashes, so strip the group id to keep the two in sync.
        return hash(get_block_hash(b.block_hash))

    def popleft(self):
        """V1 兼容接口：拿 1 个空闲块（分配 1 个新 block 用）。"""
        return self.popleft_n(1)[0]

    def popleft_n(self, n, content_hash=None):
        """分配 n 个新 block（从空闲池/驱逐中拿），V1 的 get_new_blocks 调用它。

        优先级：
          ① 先掏 hashless 池——无 hash 块复用不算"驱逐"（它没有前缀缓存身份）
          ② 再从 evictor 驱逐——拿 hashed block 等于从前缀缓存驱逐
        （原版 FreeKVCacheBlockQueue 也从队首拿，队首正是最旧的 free 块。）
        """
        ret = []
        # 优先级①：hashless 池（FIFO，popleft 从头部）
        while self._hashless_pool and len(ret) < n:
            if self.num_free_blocks == 0:
                break
            bi = self._hashless_pool.popleft()
            self.num_free_blocks -= 1
            blk = self._blocks[bi]
            if blk.ref_cnt != 0:   # 防御：不该驱逐占用中的块
                open('/tmp/ghost.log', 'a').write(
                    f'FATAL hashless bi={bi} ref_cnt={blk.ref_cnt}\n')
            ret.append(blk)
        # 优先级②：evictor 驱逐（ARC 的 T1/T2 驱逐决策）
        while len(ret) < n:
            if self.num_free_blocks == 0:
                break
            bi, _ = self.evictor.evict(content_hash=content_hash)
            if bi < 0:   # 防御：ARC 驱逐返回 -1（早期 Bug1 的防御）
                import sys
                ev = self.evictor
                print('NEG_BI', bi, 'T1_keys', list(ev.T1_table.keys())[:5], 'T2_keys', list(ev.T2_table.keys())[:5], 'T1_act', list(ev.T1_active)[:5], 'T2_act', list(ev.T2_active)[:5], 'L1q', len(ev.T1_queue), 'L2q', len(ev.T2_queue), 'nf', self.num_free_blocks, file=sys.stderr, flush=True)
            blk = self._blocks[bi]
            import sys; print(f'POP bi={bi} ref={blk.ref_cnt}', file=sys.stderr, flush=True)  # 调试日志
            if blk.ref_cnt != 0:   # 防御：驱逐了占用中的块（Bug2 场景）
                open('/tmp/ghost.log', 'a').write(
                    f'FATAL evict block={bi} ref_cnt={blk.ref_cnt} '
                    f'T1={bi in self.evictor.T1_table} T2={bi in self.evictor.T2_table}\n')
                # 清理脏状态后跳过，重试驱逐
                self.evictor.T1_table.pop(bi, None)
                self.evictor.T2_table.pop(bi, None)
                self.num_free_blocks -= 1
                continue
            self.num_free_blocks -= 1
            ret.append(blk)
        return ret

    def remove(self, block):
        """V1 兼容接口：block 被占用（touch 前缀缓存命中）时从空闲池移除。

        对应原版 FreeKVCacheBlockQueue.remove——block 出空闲池、ref_cnt 将增加。
        - ARC/LRU：直接从 T1/T2_table pop（block 不再可被驱逐）
        - DBL：调 evictor.remove() 记录 A1out/Amout 幽灵（供释放时提升 Am）
        同时记 _promote_on_release，让该 block 释放时进 T2（ARC 的 T1→T2 晋升）。
        """
        bi = block.block_id
        was_in_t1 = bi in self.evictor.T1_table
        was_in_t2 = bi in self.evictor.T2_table
        if hasattr(self.evictor, 'A1in_free_table'):
            # DBL: evictor.remove() records A1out/Amout ghost so a later
            # release via _add_to_T1/add() can promote the block to Am.
            try:
                self.evictor.remove(bi)
            except ValueError:
                pass
        else:
            self.evictor.T1_table.pop(bi, None)
            self.evictor.T2_table.pop(bi, None)
        if was_in_t1 or was_in_t2:
            # 记录：这个 block 释放时要进 T2（被 touch 过 = 有复用价值）
            self._promote_on_release[bi] = 'T1HIT' if was_in_t1 else 'T2HIT'
        self.num_free_blocks -= 1
        open('/tmp/ghost.log', 'a').write(
            f'REMOVE block={bi} wasT1={was_in_t1} wasT2={was_in_t2}\n')

    def prepend_n(self, blocks):
        """V1 兼容接口：释放无 hash 块 → 进 hashless 池（永不进 T1）。

        对应原版 free_blocks 里 blocks_without_hash → prepend_n（先驱逐）。
        原版里无 hash 块放队首、最先被复用；这里放 hashless 池（同样优先复用）。
        """
        for b in blocks:
            assert b.ref_cnt == 0, f"prepend_n block={b.block_id} ref_cnt={b.ref_cnt}"
            bi = b.block_id
            self._hashless_pool.append(bi)
            open('/tmp/ghost.log', 'a').write(f'ADD_HASHLESS block={bi}\n')
            self.num_free_blocks += 1

    def append_n(self, blocks):
        """V1 兼容接口：释放 hashed 块 → 进 evictor（T1/T2）。

        对应原版 free_blocks 里 blocks_with_hash → append_n（队尾 MRU）。
        - 若该 block 在 _promote_on_release 里（touch 命中/幽灵命中过）→ 进 T2
        - 否则进 T1
        释放顺序用单调序号 _release_seq 记录（保证 heap 顺序 = 真实释放顺序）。
        """
        for b in blocks:
            assert b.ref_cnt == 0, f"append_n block={b.block_id} ref_cnt={b.ref_cnt}"
            bi = b.block_id
            self._release_seq += 1
            # BUGFIX (8/19): append_n bypasses evictor.add(), which is the ONLY
            # caller of T1_active.remove()/T2_active.remove(). Without clearing
            # the active set here, every block that evict() marked T1_active stays
            # there forever -> T1_active saturates to max_size -> _replace short-
            # circuits -> ARC degrades to unconditional T1-FIFO (the -50pp crash).
            self.evictor.T1_active.discard(bi)
            self.evictor.T2_active.discard(bi)
            # monotonic integer key preserves the exact release order (the
            # order free_blocks passes blocks_with_hash = eviction priority),
            # mirroring the original linked-list order. time.monotonic() ties
            # within one batch and degenerates to block_id ordering.
            if bi in self._promote_on_release:
                src = self._promote_on_release.pop(bi)
                self.evictor._add_to_T2(bi, self._make_meta(b, self._release_seq))
                open('/tmp/ghost.log', 'a').write(f'ADD_T2 src={src} block={bi}\n')
            else:
                self.evictor._add_to_T1(bi, self._make_meta(b, self._release_seq))
                open('/tmp/ghost.log', 'a').write(f'ADD_T1 block={bi}\n')
            self.num_free_blocks += 1
        # BUGFIX (8/7): we bypass evictor.add(), which is the only caller of
        # _prune_ghosts() in V0. Without this, B1/B2 grow unbounded (B2 hit
        # ~55k vs B1 ~100), over-protecting T2 and hurting hit rate.
        self.evictor._prune_ghosts()

    def _make_meta(self, b, now):
        """构造 BlockMetaData（evictor 驱逐决策用的元信息）。"""
        from vllm.core.evictor import BlockMetaData
        return BlockMetaData(self._hash(b), b.block_hash_num_tokens or 0, now)

    def check_ghost_hits(self, block_hash, start=0):
        """幽灵命中检查（单 block 粒度，还原论文"逐 block 驱逐时检查"节奏）。

        由 block_pool.cache_full_blocks 在 block 写入 hash 时逐块调用。
        命中 B1 → p 上调（recency 不够）；命中 B2 → p 下调（frequency 不够）。
        delta 是比例式（与论文一致），命中后从幽灵目录删除该记录、标记待路由 T2。
        """
        h = hash(block_hash) if block_hash is not None else 0
        ev = self.evictor
        # audit: sample hashless-pool remaining size at check time (5% sample)
        self._check_count += 1
        if self._check_count % 20 == 0:
            try:
                open('/tmp/hashless.log', 'a').write(
                    f'{len(self._hashless_pool)} {int(h in ev.B1 or h in ev.B2)}\n')
            except Exception:
                pass
        if h == 0:
            return
        if h in ev.B1:
            # B1 命中：这本"被赶出去的书又来了" → recency 不够 → p 上调
            d = max(1, len(ev.B2) // max(1, len(ev.B1)))
            # C4 p-softcap: cap p-increase at p_cap (ARC_P_CAP_FRAC, default 0.5x;
            # = max_size when frac=1.0). Same as the deployed server arc_queue.py.
            ev.p = min(ev.p + d, ev.p_cap)
            del ev.B1[h]
            self._pending_t2.add(h)
            self._pending_t2_src[h] = 'B1'
            open('/tmp/ghost.log', 'a').write(
                f'GHOST B1 hit={h:x} p={ev.p} pending_t2={len(self._pending_t2)}\n')
        elif h in ev.B2:
            # B2 命中：常客书被赶走又来 → frequency 不够 → p 下调
            d = max(1, len(ev.B1) // max(1, len(ev.B2)))
            ev.p = max(ev.p - d, 0)
            del ev.B2[h]
            self._pending_t2.add(h)
            self._pending_t2_src[h] = 'B2'
            open('/tmp/ghost.log', 'a').write(
                f'GHOST B2 hit={h:x} p={ev.p} pending_t2={len(self._pending_t2)}\n')

    def on_block_hashed(self, block_hash, block_id):
        """block 真正写入 hash 时调用（block_pool.cache_full_blocks）。

        消费 check_ghost_hits 留下的 _pending_t2：该 block 释放时应路由 T2。
        （幽灵命中 → 这个内容值得进 frequency 队列保护。）
        """
        h = hash(block_hash) if block_hash is not None else 0
        if h in self._pending_t2:
            src = self._pending_t2_src.pop(h, 'B1')
            self._pending_t2.discard(h)
            self._promote_on_release[block_id] = src
            open('/tmp/ghost.log', 'a').write(
                f'CONSUME hash={h:x} src={src} block={block_id} '
                f'pending_after={len(self._pending_t2)}\n')

    def get_all_free_blocks(self):
        """V1 兼容接口：返回全部空闲块（T1 + T2 + hashless 池）。"""
        return [self._blocks[bi] for bi in self.evictor.T1_table
                if bi < len(self._blocks)] + \
               [self._blocks[bi] for bi in self.evictor.T2_table
                if bi < len(self._blocks)] + \
               [self._blocks[bi] for bi in self._hashless_pool
                if bi < len(self._blocks)]

    def iter_blocks_after(self, cursor=None):
        """V1 兼容接口：按驱逐顺序迭代空闲块。"""
        yield from self.get_all_free_blocks()
