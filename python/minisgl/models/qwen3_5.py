from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from flashinfer import chunk_gated_delta_rule
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
    get_rope,
)
from minisgl.utils import div_even, nvtx_annotate
from sgl_kernel import causal_conv1d_fwd, causal_conv1d_update

from .base import BaseLLMModel
from .utils import GatedMLP

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3_5RMSNormGated(BaseOP):
    """Gated RMSNorm used by the Gated DeltaNet out projection."""

    def __init__(self, size: int, eps: float) -> None:
        self.weight = torch.empty(size)
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        y = xf * torch.rsqrt(var + self.eps)
        y = y * self.weight.float()
        y = y * F.silu(gate.float())
        return y.to(dtype)


class Qwen3_5LinearAttention(BaseOP):
    """Qwen3.5 Gated DeltaNet linear-attention token mixer."""

    def __init__(self, config: "ModelConfig", layer_id: int) -> None:
        tp_info = get_tp_info()
        tp_size = tp_info.size
        self.layer_id = layer_id
        self.linear_layer_id = config.linear_layer_ids.index(layer_id)
        self.num_k_heads = div_even(config.linear_num_key_heads, tp_size)
        self.num_v_heads = div_even(config.linear_num_value_heads, tp_size)
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.eps = config.rms_norm_eps
        assert self.num_v_heads % self.num_k_heads == 0

        full_key_dim = config.linear_num_key_heads * self.head_k_dim
        full_value_dim = config.linear_num_value_heads * self.head_v_dim
        full_conv_dim = 2 * full_key_dim + full_value_dim

        self.in_proj_qkv = LinearColParallelMerged(
            config.hidden_size, [full_conv_dim], has_bias=False
        )
        self.in_proj_z = LinearColParallelMerged(
            config.hidden_size, [full_value_dim], has_bias=False
        )
        self.in_proj_a = LinearColParallelMerged(
            config.hidden_size, [config.linear_num_value_heads], has_bias=False
        )
        self.in_proj_b = LinearColParallelMerged(
            config.hidden_size, [config.linear_num_value_heads], has_bias=False
        )
        self.conv1d = torch.empty(self.conv_dim, 1, config.linear_conv_kernel_dim)
        self.A_log = torch.empty(self.num_v_heads)
        self.dt_bias = torch.empty(self.num_v_heads)
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.eps)
        self.out_proj = LinearRowParallel(full_value_dim, config.hidden_size, has_bias=False)

    def _gates(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        beta = torch.sigmoid(b.float())
        return g, beta

    def _forward_prefill(self, x: torch.Tensor, batch, state) -> torch.Tensor:
        qkv = self.in_proj_qkv.forward(x)
        z = self.in_proj_z.forward(x)
        a = self.in_proj_a.forward(x)
        b = self.in_proj_b.forward(x)

        # sgl-kernel's causal conv1d mutates its input in place.
        qkv_t = qkv.transpose(0, 1).contiguous()
        causal_conv1d_fwd(
            qkv_t,
            self.conv1d.squeeze(1),
            None,
            state.conv[self.linear_layer_id],
            batch.linear_cu_seqlens,
            batch.state_indices,
            batch.has_initial_state,
            True,
            -1,
        )
        conv = qkv_t.transpose(0, 1).contiguous()

        q, k, v = torch.split(conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.contiguous().view(-1, self.num_k_heads, self.head_k_dim)
        k = k.contiguous().view(-1, self.num_k_heads, self.head_k_dim)
        v = v.contiguous().view(-1, self.num_v_heads, self.head_v_dim)

        g, beta = self._gates(a, b)
        # FlashInfer's chunked GDN API takes the linear forget gate alpha,
        # while the checkpoint convention (and Qwen3-Next reference) computes
        # g in log space.
        alpha = torch.exp(g)
        core, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=alpha,
            beta=beta,
            initial_state=state.recurrent[self.linear_layer_id],
            output_state=state.recurrent[self.linear_layer_id],
            output_final_state=True,
            cu_seqlens=batch.linear_cu_seqlens,
            state_indices=batch.state_indices,
            use_qk_l2norm_in_kernel=True,
            backend="flashinfer",
        )
        core = self.norm.forward(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return core.reshape(x.shape[0], self.value_dim)

    def _forward_decode(self, x: torch.Tensor, batch, state) -> torch.Tensor:
        qkv = self.in_proj_qkv.forward(x)
        z = self.in_proj_z.forward(x)
        a = self.in_proj_a.forward(x)
        b = self.in_proj_b.forward(x)

        # The kernel mutates x in place and writes the convolution result there.
        qkv_t = qkv.unsqueeze(-1).contiguous()
        causal_conv1d_update(
            qkv_t,
            state.conv[self.linear_layer_id],
            self.conv1d.squeeze(1),
            None,
            True,
            None,
            batch.state_indices,
            -1,
        )
        conv = qkv_t.squeeze(-1)

        q, k, v = torch.split(conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.contiguous().view(-1, 1, self.num_k_heads, self.head_k_dim)
        k = k.contiguous().view(-1, 1, self.num_k_heads, self.head_k_dim)
        v = v.contiguous().view(-1, 1, self.num_v_heads, self.head_v_dim)

        core, _ = gated_delta_rule_decode_pretranspose(
            q,
            k,
            v,
            state=None,
            A_log=self.A_log.float(),
            a=a[:, None, :].contiguous(),
            dt_bias=self.dt_bias.float(),
            b=b[:, None, :].contiguous(),
            initial_state=state.recurrent[self.linear_layer_id],
            initial_state_indices=batch.state_indices,
            output_state_indices=batch.state_indices,
            use_qk_l2norm=True,
            backend="flashinfer",
        )
        core = core[:, 0]
        core = self.norm.forward(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return core.reshape(x.shape[0], self.value_dim)

    @nvtx_annotate("GDN_{}", layer_id_field="layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        state = ctx.linear_state
        assert state is not None, "Hybrid model requires a linear state pool"
        if ctx.batch.is_prefill:
            out = self._forward_prefill(x, ctx.batch, state)
        else:
            out = self._forward_decode(x, ctx.batch, state)
        return self.out_proj.forward(out)


class Qwen3_5GatedAttention(BaseOP):
    """Qwen3.5 full attention with Q+gate projection and partial RoPE."""

    def __init__(self, config: "ModelConfig", layer_id: int) -> None:
        tp_size = get_tp_info().size
        self.layer_id = layer_id
        self.kv_cache_id = config.kv_layer_ids.index(layer_id)
        self.head_dim = config.head_dim
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        full_q_dim = config.num_qo_heads * self.head_dim
        full_kv_dim = config.num_kv_heads * self.head_dim

        self.q_proj = LinearColParallelMerged(config.hidden_size, [2 * full_q_dim], has_bias=False)
        self.k_proj = LinearColParallelMerged(config.hidden_size, [full_kv_dim], has_bias=False)
        self.v_proj = LinearColParallelMerged(config.hidden_size, [full_kv_dim], has_bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary = get_rope(
            head_dim=config.rotary_config.head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items())
                if config.rotary_config.scaling
                else None
            ),
        )
        self.o_proj = LinearOProj(full_q_dim, config.hidden_size, has_bias=False)

    @nvtx_annotate("MHA_gated_{}", layer_id_field="layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        tokens = x.shape[0]

        q_and_gate = self.q_proj.forward(x).view(-1, self.num_qo_heads, 2 * self.head_dim)
        q, gate = torch.chunk(q_and_gate, 2, dim=-1)
        q = self.q_norm.forward(q.contiguous())
        gate = gate.contiguous()

        k = self.k_proj.forward(x).view(-1, self.num_kv_heads, self.head_dim).contiguous()
        k = self.k_norm.forward(k)
        v = self.v_proj.forward(x).view(-1, self.num_kv_heads * self.head_dim).contiguous()

        q, k = self.rotary.forward(
            batch.positions,
            q.reshape(tokens, self.num_qo_heads * self.head_dim),
            k.reshape(tokens, self.num_kv_heads * self.head_dim),
        )
        q = q.view(tokens, self.num_qo_heads, self.head_dim)

        o = ctx.attn_backend.forward(q, k, v, self.kv_cache_id, batch)
        o = o.reshape(tokens, self.num_qo_heads * self.head_dim)
        o = o * torch.sigmoid(gate.reshape(o.shape)).to(o.dtype)
        return self.o_proj.forward(o)


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int):
        self._layer_id = layer_id
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.is_linear = config.layer_types[layer_id] == "linear_attention"
        if self.is_linear:
            self.linear_attn = Qwen3_5LinearAttention(config, layer_id)
        else:
            self.self_attn = Qwen3_5GatedAttention(config, layer_id)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = GatedMLP(config)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.linear_attn.forward(x) if self.is_linear else self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: "ModelConfig"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig"):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=(self.model.embed_tokens if config.tie_word_embeddings else None),
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Qwen3_5ForCausalLM"]
