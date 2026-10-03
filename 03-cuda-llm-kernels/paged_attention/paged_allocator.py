import torch


class PagedBlockAllocator:
    """Physical page frame allocator for KV cache blocks.

    Implements O(1) stack-based allocation and deallocation (Kwon et al., 2023).
    """

    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks: list[int] = list(range(total_blocks - 1, -1, -1))

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate(self, num_blocks: int) -> list[int]:
        if num_blocks > len(self.free_blocks):
            raise MemoryError(
                f"Out of KV cache memory: requested {num_blocks} blocks, "
                f"only {len(self.free_blocks)} available."
            )
        allocated = [self.free_blocks.pop() for _ in range(num_blocks)]
        return allocated

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            self.free_blocks.append(bid)


class SequenceBlockTableManager:
    """Virtual-to-physical block table manager for active inference sequences."""

    def __init__(self, allocator: PagedBlockAllocator, block_size: int = 16):
        self.allocator = allocator
        self.block_size = block_size
        self.seq_to_blocks: dict[int, list[int]] = {}
        self.seq_lengths: dict[int, int] = {}

    def allocate_sequence(self, seq_id: int, initial_tokens: int) -> None:
        num_blocks = (initial_tokens + self.block_size - 1) // self.block_size
        self.seq_to_blocks[seq_id] = self.allocator.allocate(num_blocks)
        self.seq_lengths[seq_id] = initial_tokens

    def append_token(self, seq_id: int) -> None:
        curr_len = self.seq_lengths[seq_id]
        if curr_len % self.block_size == 0:
            new_block = self.allocator.allocate(1)[0]
            self.seq_to_blocks[seq_id].append(new_block)
        self.seq_lengths[seq_id] = curr_len + 1

    def free_sequence(self, seq_id: int) -> None:
        if seq_id in self.seq_to_blocks:
            self.allocator.free(self.seq_to_blocks[seq_id])
            del self.seq_to_blocks[seq_id]
            del self.seq_lengths[seq_id]

    def build_block_table_tensor(self, seq_ids: list[int], max_blocks: int) -> torch.Tensor:
        batch_size = len(seq_ids)
        table = torch.full((batch_size, max_blocks), fill_value=-1, dtype=torch.int32)
        for i, sid in enumerate(seq_ids):
            blocks = self.seq_to_blocks[sid]
            n = min(len(blocks), max_blocks)
            table[i, :n] = torch.tensor(blocks[:n], dtype=torch.int32)
        return table
