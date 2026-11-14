# Qwen3.8-27B (Qwen3.5 hybrid) usage notes

Mini-SGLang supports the text backbone of Qwen3.8-27B / Qwen3.5-style hybrid models.

## Architecture

- 64 decoder layers
- 48 Gated DeltaNet linear-attention layers
- 16 full-attention layers (indices 3, 7, ..., 63)
- Full attention: 24 Q heads / 4 KV heads, head_dim 256, output gate, QK RMSNorm, partial RoPE (64/256)
- Linear attention: 16 K heads / 48 V heads, head_dim 128, depthwise conv1d kernel 4; recurrent state in float32
- Text-only: the checkpoint's `model.visual.*` vision tower and `mtp.*` weights are skipped

## Launch

```bash
python -m minisgl \
  --model /path/to/Qwen3.8-27B \
  --tp 1 \
  --attn fi \
  --cache naive \
  --max-running-requests 32 \
  --max-prefill-length 1024 \
  --cuda-graph-max-bs 8
```

Notes:

- `--attn fi` is the recommended first choice on SM120/Blackwell; `auto` also chooses it there.
- `--cache naive` is forced for hybrid models; radix prefix cache is not supported yet and will be downgraded with a warning.
- `--max-running-requests` is an upper bound. Mini-SGLang caps it further based on the linear-attention state pool memory.
- Start with a moderate `--cuda-graph-max-bs` (e.g. 8 or 16); prefill CUDA graph capture for many shapes can be expensive.
- `--max-seq-len-override` and `--num-pages` are useful for bounded-memory tests.

## Memory

At TP=1, one linear-attention state slot is approximately:

```text
48 layers * 48 value heads * 128 * 128 * 4 B  ≈ 144 MiB
+ conv state                                  ≈   3 MiB
≈ 147 MiB per request
```

The engine reserves state memory before sizing the paged KV cache and caps
`max_running_req` accordingly. With 96 GiB and bf16 weights, a starting point of
16–32 running requests is reasonable; tune from the startup log:

```text
Linear state: ... MiB/slot, ... GiB for N slots
Allocating ... tokens for KV cache, K + V = ... GiB
```

## Known limitations

- No image/video input yet.
- No MTP / speculative decoding.
- No radix prefix cache for the hybrid model yet (naive cache only).
- `linear_attn` state is float32; a bf16 state option is not exposed yet.
- The checkpoint's `language_model_only` flag is ignored by the model loader; only the text weights are loaded.

## Validation

The implementation was checked against SGLang 0.5.23 on the same Qwen3.8-27B
weights and GPU: greedy chat output for the first 64 generation tokens matched
SGLang. CUDA Graph and chunked-prefill runs also produced the same token ids.
