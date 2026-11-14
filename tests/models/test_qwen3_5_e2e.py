from __future__ import annotations

import os

import pytest
import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM

MODEL_PATH = os.environ.get("MINISGL_QWEN35_MODEL", "/root/autodl-tmp/model")
RUN_E2E = os.environ.get("MINISGL_RUN_QWEN35_E2E") == "1"

# Greedy first eight tokens for the fixed chat prompt below, cross-checked with
# SGLang 0.5.23 on the same checkpoint.  This catches norm/gate/state regressions.
EXPECTED_PREFIX = [97237, 96719, 95826, 110827, 98682, 108829, 113672, 119721]


@pytest.mark.skipif(
    not RUN_E2E or not os.path.isdir(MODEL_PATH) or not torch.cuda.is_available(),
    reason="set MINISGL_RUN_QWEN35_E2E=1 and provide the Qwen3.8 weights/CUDA",
)
def test_qwen3_5_chunked_prefill_and_graph() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话解释什么是量子纠缠。"}],
        add_generation_prompt=True,
        tokenize=True,
    )
    llm = LLM(
        MODEL_PATH,
        dtype=torch.bfloat16,
        max_running_req=1,
        cuda_graph_bs=[1],
        num_page_override=4096,
        max_seq_len_override=4096,
        max_extend_tokens=16,
    )
    output = llm.generate(
        [input_ids], SamplingParams(temperature=0.0, max_tokens=8)
    )[0]
    assert output["token_ids"] == EXPECTED_PREFIX
