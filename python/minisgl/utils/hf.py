from __future__ import annotations

import copy
import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


class _AttrDict(dict):
    """Lightweight config object for architectures unknown to transformers.

    New model releases often require a newer transformers version than the one
    installed.  The inference code only needs attribute-style access to a small
    subset of the config, so fall back to a plain JSON-backed mapping instead of
    forcing users to upgrade transformers.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return _AttrDict({k: _to_attr_dict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(v) for v in value]
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_path)
    except Exception:
        # Fallback for model architectures which are newer than the installed
        # transformers release (e.g. Qwen3.5 with transformers 4.57.x).
        config_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_path):
            raise
        with open(config_path, "r", encoding="utf-8") as f:
            return _to_attr_dict(json.load(f))


def cached_load_hf_config(model_path: str) -> Any:
    config = _load_hf_config(model_path)
    if isinstance(config, dict):
        return _to_attr_dict(copy.deepcopy(_to_plain(config)))
    return type(config)(**config.to_dict())


def get_hf_dtype(config: Any) -> str | None:
    dtype = getattr(config, "dtype", None)
    if dtype is None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            dtype = getattr(text_config, "dtype", None)
    return dtype if isinstance(dtype, str) else None


def load_eos_token_ids(model_path: str, tokenizer: PreTrainedTokenizerBase) -> set[int]:
    eos_ids: set[int] = set()

    generation_config_path = os.path.join(model_path, "generation_config.json")
    if os.path.isfile(generation_config_path):
        try:
            with open(generation_config_path, "r", encoding="utf-8") as f:
                eos = json.load(f).get("eos_token_id")
            if isinstance(eos, int):
                eos_ids.add(eos)
            elif isinstance(eos, list):
                eos_ids.update(int(x) for x in eos)
        except Exception:
            pass

    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_eos, int):
        eos_ids.add(tokenizer_eos)
    elif isinstance(tokenizer_eos, list):
        eos_ids.update(int(x) for x in tokenizer_eos)

    return eos_ids


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
