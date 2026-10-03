import pytest
import torch

from paged_attention.paged_allocator import PagedBlockAllocator, SequenceBlockTableManager


def test_allocator_basic_lifecycle():
    allocator = PagedBlockAllocator(total_blocks=10, block_size=16)
    assert allocator.get_num_free_blocks() == 10

    blocks = allocator.allocate(3)
    assert len(blocks) == 3
    assert allocator.get_num_free_blocks() == 7

    allocator.free(blocks[:1])
    assert allocator.get_num_free_blocks() == 8


def test_allocator_out_of_memory():
    allocator = PagedBlockAllocator(total_blocks=2, block_size=16)
    allocator.allocate(2)
    assert allocator.get_num_free_blocks() == 0

    with pytest.raises(MemoryError):
        allocator.allocate(1)


def test_sequence_manager_expansion():
    allocator = PagedBlockAllocator(total_blocks=100, block_size=16)
    manager = SequenceBlockTableManager(allocator, block_size=16)

    manager.allocate_sequence(seq_id=1, initial_tokens=16)
    assert len(manager.seq_to_blocks[1]) == 1

    # Appending 17th token triggers new block allocation
    manager.append_token(seq_id=1)
    assert len(manager.seq_to_blocks[1]) == 2
    assert manager.seq_lengths[1] == 17

    table_tensor = manager.build_block_table_tensor([1], max_blocks=4)
    assert table_tensor.shape == (1, 4)
    assert table_tensor.dtype == torch.int32
    assert table_tensor[0, 0].item() == manager.seq_to_blocks[1][0]
    assert table_tensor[0, 1].item() == manager.seq_to_blocks[1][1]


def test_sequence_manager_free():
    allocator = PagedBlockAllocator(total_blocks=10, block_size=16)
    manager = SequenceBlockTableManager(allocator, block_size=16)

    manager.allocate_sequence(seq_id=42, initial_tokens=32)
    assert allocator.get_num_free_blocks() == 8

    manager.free_sequence(seq_id=42)
    assert allocator.get_num_free_blocks() == 10
    assert 42 not in manager.seq_to_blocks
    assert 42 not in manager.seq_lengths
