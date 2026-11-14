from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    # Hybrid attention (Qwen3.5 Gated DeltaNet) metadata.  Defaults keep all
    # existing dense models on the full-attention path.
    layer_types: Tuple[str, ...] = ()
    partial_rotary_factor: float = 1.0
    attn_output_gate: bool = False
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 0
    mamba_ssm_dtype: str = "float32"

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @property
    def num_kv_layers(self) -> int:
        return sum(t == "full_attention" for t in self.layer_types)

    @property
    def kv_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "full_attention")

    @property
    def linear_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "linear_attention")

    @property
    def num_linear_layers(self) -> int:
        return len(self.linear_layer_ids)

    @property
    def is_hybrid(self) -> bool:
        return any(t == "linear_attention" for t in self.layer_types)

    @property
    def supports_prefix_cache(self) -> bool:
        # A linear-attention (Gated DeltaNet) model needs its recurrent state
        # at a matched prefix.  The current radix cache only stores token ids
        # and KV pages, so prefix reuse is not supported for hybrid models yet.
        return not self.is_hybrid

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if getattr(config, "text_config", None) is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling", "rope_parameters"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = list(getattr(config, "architectures", ["LlamaForCausalLM"]))

        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention"] * num_layers
        layer_types = tuple(layer_types)

        rope_scaling = getattr(config, "rope_scaling", None)
        rope_parameters = getattr(config, "rope_parameters", None)
        rope_dict = rope_parameters or rope_scaling
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None and isinstance(rope_dict, dict):
            rope_theta = rope_dict.get("rope_theta")
        if rope_theta is None:
            rope_theta = 10000.0

        partial_rotary_factor = float(
            getattr(
                config,
                "partial_rotary_factor",
                rope_dict.get("partial_rotary_factor", 1.0) if isinstance(rope_dict, dict) else 1.0,
            )
        )
        rotary_dim = int(head_dim * partial_rotary_factor)
        assert (
            0 < rotary_dim <= head_dim
        ), f"Invalid rotary_dim={rotary_dim} for head_dim={head_dim}"

        # rope_parameters (new format) is only passed through for non-default
        # rope types; plain Qwen3.5 text uses standard 1-D RoPE.
        scaling = rope_scaling
        if scaling is None and isinstance(rope_parameters, dict):
            if rope_parameters.get("rope_type", "default") != "default":
                scaling = rope_parameters

        return cls(
            num_layers=num_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=float(rope_theta),
                scaling=scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            layer_types=layer_types,
            partial_rotary_factor=partial_rotary_factor,
            attn_output_gate=bool(getattr(config, "attn_output_gate", False)),
            linear_num_key_heads=int(getattr(config, "linear_num_key_heads", 0)),
            linear_num_value_heads=int(getattr(config, "linear_num_value_heads", 0)),
            linear_key_head_dim=int(getattr(config, "linear_key_head_dim", 0)),
            linear_value_head_dim=int(getattr(config, "linear_value_head_dim", 0)),
            linear_conv_kernel_dim=int(getattr(config, "linear_conv_kernel_dim", 0)),
            mamba_ssm_dtype=getattr(config, "mamba_ssm_dtype", "float32"),
        )
