from __future__ import annotations

import json
import os

import pytest
import torch

from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import ModelConfig, create_model
from minisgl.models.weight import _normalize_key
from minisgl.utils import cached_load_hf_config, torch_dtype

MODEL_PATH = os.environ.get("MINISGL_QWEN35_MODEL", "/root/autodl-tmp/model")


@pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH) or not torch.cuda.is_available(),
    reason="Qwen3.5 model directory / CUDA not available",
)
def test_qwen3_5_checkpoint_keys_cover_runtime_model() -> None:
    set_tp_info(0, 1)
    set_rope_device(torch.device("cuda"))
    config = ModelConfig.from_hf(cached_load_hf_config(MODEL_PATH))

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(config)
    runtime_keys = set(model.state_dict())

    with open(os.path.join(MODEL_PATH, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    checkpoint_keys: set[str] = set()
    for raw_name in weight_map:
        name = _normalize_key(raw_name)
        if name is None:
            continue
        if name.endswith("linear_attn.conv1d.weight"):
            name = name.removesuffix(".weight")
        checkpoint_keys.add(name)

    # MLP gate/up are the only projections merged at load time.
    checkpoint_keys = {
        key.replace(".gate_proj.", ".gate_up_proj.").replace(
            ".up_proj.", ".gate_up_proj."
        )
        for key in checkpoint_keys
    }

    assert runtime_keys == checkpoint_keys
