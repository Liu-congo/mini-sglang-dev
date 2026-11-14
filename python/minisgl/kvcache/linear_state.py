from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even


class LinearStatePool:
    """Per-request recurrent and convolution state for hybrid linear attention.

    The pool rows are addressed by ``table_idx`` (one slot per running
    request).  Both tensors are layer-major so indexing one layer yields a
    dense ``[slots, ...]`` tensor, which is what FlashInfer GDN and sgl-kernel
    causal-conv1d expect.
    """

    def __init__(
        self,
        num_slots: int,
        num_layers: int,
        num_v_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_dim: int,
        conv_kernel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # FlashInfer GDN uses V-major state: [H, V, K].
        self.recurrent = torch.zeros(
            (num_layers, num_slots, num_v_heads, head_v_dim, head_k_dim),
            dtype=torch.float32,
            device=device,
        )
        # sgl-kernel causal_conv1d expects [slots, channels, kernel - 1].
        self.conv = torch.zeros(
            (num_layers, num_slots, conv_dim, conv_kernel - 1),
            dtype=dtype,
            device=device,
        )
        self.device = device
        self.dtype = dtype

    @property
    def num_slots(self) -> int:
        return self.recurrent.shape[1]

    @property
    def num_layers(self) -> int:
        return self.recurrent.shape[0]

    @classmethod
    def dimensions(cls, model_config, tp_size: int) -> tuple[int, int, int, int, int]:
        num_k_heads = div_even(model_config.linear_num_key_heads, tp_size)
        num_v_heads = div_even(model_config.linear_num_value_heads, tp_size)
        key_dim = num_k_heads * model_config.linear_key_head_dim
        value_dim = num_v_heads * model_config.linear_value_head_dim
        conv_dim = 2 * key_dim + value_dim
        return (
            num_v_heads,
            model_config.linear_value_head_dim,
            model_config.linear_key_head_dim,
            conv_dim,
            model_config.linear_conv_kernel_dim,
        )

    @classmethod
    def bytes_per_slot(cls, model_config, dtype: torch.dtype, tp_size: int) -> int:
        if not model_config.is_hybrid:
            return 0
        num_v_heads, head_v_dim, head_k_dim, conv_dim, conv_kernel = cls.dimensions(
            model_config, tp_size
        )
        recurrent_bytes = (
            model_config.num_linear_layers
            * num_v_heads
            * head_v_dim
            * head_k_dim
            * 4  # float32 state
        )
        conv_bytes = model_config.num_linear_layers * conv_dim * (conv_kernel - 1) * dtype.itemsize
        return recurrent_bytes + conv_bytes

    @classmethod
    def from_model_config(
        cls,
        model_config,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "LinearStatePool":
        tp_size = get_tp_info().size
        num_v_heads, head_v_dim, head_k_dim, conv_dim, conv_kernel = cls.dimensions(
            model_config, tp_size
        )
        return cls(
            num_slots=num_slots,
            num_layers=model_config.num_linear_layers,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            conv_dim=conv_dim,
            conv_kernel=conv_kernel,
            dtype=dtype,
            device=device,
        )

    def reset(self, slot: int) -> None:
        """Reset one reusable request slot before it is assigned to a new request."""
        self.recurrent[:, slot].zero_()
        self.conv[:, slot].zero_()
