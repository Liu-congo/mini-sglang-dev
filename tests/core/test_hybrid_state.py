from __future__ import annotations

import pytest
import torch

from minisgl.kvcache import LinearStatePool
from minisgl.scheduler.table import TableManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_linear_state_pool_reset() -> None:
    pool = LinearStatePool(
        num_slots=2,
        num_layers=2,
        num_v_heads=2,
        head_v_dim=8,
        head_k_dim=8,
        conv_dim=24,
        conv_kernel=4,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    pool.recurrent[:, 1].fill_(1.0)
    pool.conv[:, 1].fill_(1.0)
    pool.recurrent[:, 0].fill_(2.0)

    pool.reset(1)

    assert torch.count_nonzero(pool.recurrent[:, 1]) == 0
    assert torch.count_nonzero(pool.conv[:, 1]) == 0
    assert torch.all(pool.recurrent[:, 0] == 2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_table_manager_resets_allocated_state() -> None:
    page_table = torch.zeros((3, 4), dtype=torch.int32, device="cuda")
    pool = LinearStatePool(
        num_slots=3,
        num_layers=1,
        num_v_heads=1,
        head_v_dim=8,
        head_k_dim=8,
        conv_dim=8,
        conv_kernel=4,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    pool.recurrent[:, 2].fill_(3.0)
    manager = TableManager(2, page_table, pool)

    slot = manager.allocate()

    assert slot == 1
    assert torch.count_nonzero(pool.recurrent[:, slot]) == 0
    assert torch.count_nonzero(pool.conv[:, slot]) == 0
