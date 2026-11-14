from __future__ import annotations

from types import SimpleNamespace

from minisgl.models import ModelConfig


def test_qwen3_5_hybrid_config() -> None:
    text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        num_hidden_layers=4,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        hidden_size=5120,
        vocab_size=248320,
        intermediate_size=17408,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=262144,
        tie_word_embeddings=False,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        partial_rotary_factor=0.25,
        attn_output_gate=True,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        },
    )
    config = SimpleNamespace(
        text_config=text_config,
        architectures=["Qwen3_5ForConditionalGeneration"],
    )

    result = ModelConfig.from_hf(config)

    assert result.num_layers == 4
    assert result.num_kv_layers == 1
    assert result.kv_layer_ids == (3,)
    assert result.linear_layer_ids == (0, 1, 2)
    assert result.num_linear_layers == 3
    assert result.is_hybrid
    assert not result.supports_prefix_cache
    assert result.rotary_config.rotary_dim == 64
    assert result.rotary_config.base == 10_000_000
    assert result.attn_output_gate
    assert result.linear_num_key_heads == 16
    assert result.linear_num_value_heads == 48


def test_dense_config_defaults_to_full_attention() -> None:
    config = SimpleNamespace(
        model_type="llama",
        num_hidden_layers=2,
        num_attention_heads=8,
        hidden_size=128,
        vocab_size=256,
        intermediate_size=256,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        rope_theta=10_000,
        rope_scaling=None,
        architectures=["LlamaForCausalLM"],
    )

    result = ModelConfig.from_hf(config)

    assert result.layer_types == ("full_attention", "full_attention")
    assert result.num_kv_layers == 2
    assert not result.is_hybrid
    assert result.supports_prefix_cache
    assert result.rotary_config.rotary_dim == result.head_dim
