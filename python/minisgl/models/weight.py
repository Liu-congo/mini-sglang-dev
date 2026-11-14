from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _shard_qwen3_5_tensor(key: str, value: torch.Tensor, r: int, n: int, config) -> torch.Tensor:
    """Qwen3.5-specific sharding for GDN projections and the gated q_proj."""
    if key.endswith(("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight")):
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        q = value[:key_dim].chunk(n, dim=0)[r]
        k = value[key_dim : 2 * key_dim].chunk(n, dim=0)[r]
        v = value[2 * key_dim :].chunk(n, dim=0)[r]
        return torch.cat([q, k, v], dim=0).clone()

    if key.endswith(
        (
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
        )
    ):
        return value.chunk(n, dim=0)[r].clone()

    if key.endswith(("linear_attn.A_log", "linear_attn.dt_bias")):
        return value.chunk(n, dim=0)[r].clone()

    if key.endswith("linear_attn.out_proj.weight"):
        return value.chunk(n, dim=1)[r].clone()

    return _shard_tensor(key, value, r, n, config.num_kv_heads)


def _normalize_key(name: str) -> str | None:
    """Strip multimodal wrappers and drop weights that are out of scope."""
    if name.startswith(
        (
            "model.visual.",
            "visual.",
            "mtp.",
            "vision_tower.",
            "multi_modal_projector.",
        )
    ):
        return None
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model.") :]
    if name.startswith("language_model."):
        return name[len("language_model.") :]
    return name


def _needs_add_one(key: str) -> bool:
    # Qwen3.5 (like Qwen3-Next) stores these RMSNorm weights as zero-centered
    # values and computes (1 + weight) at runtime.  Convert once at load time
    # so the existing FlashInfer RMSNorm kernels remain usable.
    if "linear_attn.norm" in key:
        return False
    return (
        key.endswith("input_layernorm.weight")
        or key.endswith("post_attention_layernorm.weight")
        or key.endswith("self_attn.q_norm.weight")
        or key.endswith("self_attn.k_norm.weight")
        or key == "model.norm.weight"
    )


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()
    is_qwen3_5 = config.model_type == "qwen3_5_text" or any(
        arch.startswith("Qwen3_5") for arch in config.architectures
    )

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _normalize_key(raw_name)
                if name is None:
                    continue
                raw = f.get_tensor(raw_name)
                if is_qwen3_5:
                    tensor = _shard_qwen3_5_tensor(name, raw, tp_info.rank, tp_info.size, config)
                else:
                    tensor = _shard_tensor(
                        name, raw, tp_info.rank, tp_info.size, config.num_kv_heads
                    )
                del raw

                if is_qwen3_5 and _needs_add_one(name):
                    tensor = (tensor.float() + 1.0).to(tensor.dtype)

                if is_qwen3_5 and name.endswith("linear_attn.conv1d.weight"):
                    # The runtime Conv1D parameter is a bare tensor named
                    # ``linear_attn.conv1d``; keep BaseOP.state_dict keys aligned.
                    name = name.removesuffix(".weight")

                info = _get_merge_info(name)
                if is_qwen3_5 and any(
                    suffix in name for suffix in (".q_proj", ".k_proj", ".v_proj")
                ):
                    # Qwen3.5 full attention keeps q/k/v separate because
                    # q_proj also contains the output gate.
                    info = None

                if info is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
