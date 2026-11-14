from __future__ import annotations

import pytest
import torch

from minisgl.layers.rotary import RotaryEmbedding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_partial_rope_matches_reference() -> None:
    head_dim, rotary_dim, positions_n = 256, 64, 16
    with torch.device("cuda"):
        rope = RotaryEmbedding(head_dim, rotary_dim, positions_n, 10_000.0)
    q = torch.randn(5, 2 * head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(5, 1 * head_dim, device="cuda", dtype=torch.bfloat16)
    q_ref, k_ref = q.clone(), k.clone()
    pos = torch.arange(5, device="cuda", dtype=torch.int64)

    rope.forward(pos, q, k)
    torch.cuda.synchronize()

    cache = rope._cos_sin_cache
    cos, sin = cache[pos, : rotary_dim // 2], cache[pos, rotary_dim // 2 :]
    cos = cos[:, None, :]
    sin = sin[:, None, :]

    def ref(x: torch.Tensor) -> torch.Tensor:
        x = x.view(5, -1, head_dim)
        x1, x2 = x[..., : rotary_dim // 2], x[..., rotary_dim // 2 : rotary_dim]
        rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([rot, x[..., rotary_dim:]], dim=-1)

    assert torch.allclose(
        q.view(5, 2, head_dim).float(), ref(q_ref).float(), atol=2e-2, rtol=2e-2
    )
    assert torch.allclose(
        k.view(5, 1, head_dim).float(), ref(k_ref).float(), atol=2e-2, rtol=2e-2
    )
    # The non-rotary tail must remain untouched.
    assert torch.equal(q.view(5, 2, head_dim)[..., rotary_dim:], q_ref.view(5, 2, head_dim)[..., rotary_dim:])
    assert torch.equal(k.view(5, 1, head_dim)[..., rotary_dim:], k_ref.view(5, 1, head_dim)[..., rotary_dim:])
