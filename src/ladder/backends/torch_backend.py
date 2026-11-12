"""Rungs 00 and 01: the model run with plain PyTorch, float32, on CPU.

The greedy loop is written by hand instead of calling ``model.generate`` so that
the KV-cache is visible in the code and every token gets an exact timestamp.

The tokenizer is only used through two calls, so anything that provides them
works (a Hugging Face tokenizer, or a stand-in in tests):

* ``tokenizer.encode(text, add_special_tokens=False) -> list[int]``
* ``tokenizer.decode(token_ids) -> str``
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ladder.backends.base import Backend, BackendInfo, GenerationResult

# Files that hold model weights in a Hugging Face model directory.
WEIGHT_PATTERNS = ("*.safetensors", "*.bin")


class TorchBackend(Backend):
    """PyTorch eager-mode backend, with or without the KV-cache."""

    def __init__(
        self,
        model_id: str,
        rung: str,
        use_cache: bool = True,
        num_threads: int | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        """Load the model, or use an injected one.

        Args:
            model_id: Hub id or local directory. Used for loading unless
                ``model`` and ``tokenizer`` are given, and always recorded.
            rung: Rung name recorded in ``info``.
            use_cache: Reuse keys and values between steps (rung 01) or
                recompute the whole sequence every step (rung 00).
            num_threads: Passed to ``torch.set_num_threads`` when given.
            model, tokenizer: Already-built objects, so tests can inject a tiny
                random model without any download.
        """
        # Libraries are imported at module level, so library import time is not
        # counted as model load time.
        start = time.perf_counter()
        if num_threads is not None:
            torch.set_num_threads(num_threads)

        injected = model is not None and tokenizer is not None
        if not injected:
            # transformers 5 defaults to the dtype in the model config
            # (bfloat16 for SmolLM2), so float32 has to be requested explicitly.
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        if model.dtype != torch.float32:
            raise ValueError(f"TorchBackend expects a float32 model, got {model.dtype}")

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.use_cache = use_cache
        load_time_s = time.perf_counter() - start

        self._info = BackendInfo(
            rung=rung,
            model_id=model_id,
            engine="pytorch",
            precision="fp32",
            size_on_disk_bytes=0 if injected else weights_size_bytes(model_id),
            load_time_s=load_time_s,
            num_threads=torch.get_num_threads(),
            extra={"use_cache": str(use_cache)},
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def logits(self, input_ids: np.ndarray) -> np.ndarray:
        ids = torch.from_numpy(np.asarray(input_ids, dtype=np.int64))
        with torch.inference_mode():
            out = self.model(input_ids=ids, use_cache=False)
        return out.logits.float().numpy()

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> GenerationResult:
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        start = time.perf_counter()
        token_ids: list[int] = []
        token_times_s: list[float] = []
        # Shape (1, seq): one sequence, no padding, so no attention mask needed.
        sequence = torch.tensor([prompt_ids], dtype=torch.long)
        past = None

        with torch.inference_mode():
            for step in range(max_new_tokens):
                if not self.use_cache:
                    # Rung 00: recompute every position, every step.
                    out = self.model(input_ids=sequence, use_cache=False)
                elif step == 0:
                    # First step: the whole prompt goes in; keep its keys/values.
                    out = self.model(input_ids=sequence, use_cache=True)
                else:
                    # Later steps: only the newest token, plus the cache.
                    out = self.model(
                        input_ids=sequence[:, -1:], past_key_values=past, use_cache=True
                    )
                past = out.past_key_values if self.use_cache else None

                next_id = int(out.logits[0, -1].argmax())
                token_times_s.append(time.perf_counter() - start)
                token_ids.append(next_id)
                sequence = torch.cat([sequence, torch.tensor([[next_id]])], dim=1)

        return GenerationResult(
            token_ids=token_ids, token_times_s=token_times_s, prompt_tokens=len(prompt_ids)
        )


def weights_size_bytes(model_id: str) -> int:
    """Total size of the weight files for ``model_id``, or 0 if not found locally.

    Looks in ``model_id`` if it is a directory, otherwise in the local Hugging
    Face cache. Never downloads anything.
    """
    directory = Path(model_id)
    if not directory.is_dir():
        try:
            from huggingface_hub import snapshot_download

            directory = Path(snapshot_download(model_id, local_files_only=True))
        except Exception:  # not cached, or huggingface_hub unavailable
            return 0
    # stat() follows the symlinks the HF cache uses, so this is the real size.
    return sum(p.stat().st_size for pattern in WEIGHT_PATTERNS for p in directory.glob(pattern))
