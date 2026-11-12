"""Shared fixtures: a tiny random GPT-2 and a real tokenizer, built offline.

Nothing here downloads anything. The tokenizer is a byte-level BPE trained on a
few sentences with the ``tokenizers`` library, wrapped in transformers'
``PreTrainedTokenizerFast``, so it behaves like a real Hugging Face tokenizer
(including ``save_pretrained``) and can encode any text.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from ladder.backends.torch_backend import TorchBackend

EOS = "<|endoftext|>"
TRAINING_TEXT = [
    "The quick brown fox jumps over the lazy dog.",
    "A language model predicts the next token from the tokens before it.",
    "The cache stores keys and values so they are not computed twice.",
]


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Byte-level BPE: the 256 byte symbols plus a few learned merges."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[EOS],
        show_progress=False,
    )
    tok.train_from_iterator(TRAINING_TEXT, trainer=trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS)


def build_model(vocab_size: int, eos_id: int, seed: int) -> GPT2LMHeadModel:
    """Two-layer GPT-2 with random weights; the seed makes it reproducible."""
    torch.manual_seed(seed)
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=vocab_size,
        n_positions=256,
        bos_token_id=eos_id,
        eos_token_id=eos_id,
    )
    return GPT2LMHeadModel(config).eval()


@pytest.fixture(scope="session")
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    return build_tokenizer()


@pytest.fixture(scope="session")
def tiny_model_factory(tiny_tokenizer) -> Callable[[int], GPT2LMHeadModel]:
    """Build a tiny model with a given seed (different seeds, different weights)."""

    def make(seed: int = 0) -> GPT2LMHeadModel:
        return build_model(len(tiny_tokenizer), tiny_tokenizer.eos_token_id, seed)

    return make


@pytest.fixture(scope="session")
def tiny_model_and_tokenizer(tiny_model_factory, tiny_tokenizer):
    return tiny_model_factory(0), tiny_tokenizer


@pytest.fixture
def tiny_backend(tiny_model_and_tokenizer) -> TorchBackend:
    model, tokenizer = tiny_model_and_tokenizer
    return TorchBackend("tiny", rung="01_torch_fp32", model=model, tokenizer=tokenizer)


@pytest.fixture
def tiny_nocache_backend(tiny_model_and_tokenizer) -> TorchBackend:
    model, tokenizer = tiny_model_and_tokenizer
    return TorchBackend(
        "tiny", rung="00_torch_nocache", use_cache=False, model=model, tokenizer=tokenizer
    )


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, tiny_model_and_tokenizer):
    """The tiny model and tokenizer saved like a hub checkpoint, for loading by path."""
    model, tokenizer = tiny_model_and_tokenizer
    directory = tmp_path_factory.mktemp("tiny-model")
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory
