import numpy as np
import pytest

from ladder.backends.base import Backend, BackendInfo, GenerationResult
from ladder.backends.torch_backend import TorchBackend
from ladder.bench.quality import (
    evaluate_quality,
    kl_divergence,
    load_text_file,
    log_softmax64,
    make_windows,
)


class UniformBackend(Backend):
    """Every token equally likely: perplexity must be exactly the vocab size."""

    vocab_size = 50

    @property
    def info(self) -> BackendInfo:
        return BackendInfo("uniform", "fake", "fake", "fp32", 0, 0.0, 1)

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    def logits(self, input_ids: np.ndarray) -> np.ndarray:
        return np.zeros((*input_ids.shape, self.vocab_size), dtype=np.float32)

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> GenerationResult:
        raise NotImplementedError


def test_make_windows_drops_tail_and_respects_max():
    w = make_windows(list(range(10)), window=3)
    assert w.shape == (3, 3)
    assert w.dtype == np.int64
    assert w[-1].tolist() == [6, 7, 8]
    assert make_windows(list(range(10)), window=3, max_windows=2).shape == (2, 3)
    with pytest.raises(ValueError):
        make_windows([1, 2], window=3)


def test_uniform_perplexity_equals_vocab_size():
    windows = make_windows(np.random.default_rng(0).integers(0, 50, 200), window=20)
    result = evaluate_quality(UniformBackend(), windows)
    assert result.perplexity == pytest.approx(50, rel=1e-9)
    assert result.top1_agreement is None
    assert result.mean_kl is None
    assert (result.num_windows, result.num_tokens) == (10, 200)


def test_log_softmax64_is_stable_and_normalised():
    logp = log_softmax64(np.array([[1000.0, 1000.0], [0.0, -1000.0]], dtype=np.float32))
    assert logp.dtype == np.float64
    assert np.all(np.isfinite(logp[0]))
    np.testing.assert_allclose(np.exp(logp[0]).sum(), 1.0)
    np.testing.assert_allclose(logp[0], np.log([0.5, 0.5]))


def test_kl_known_value():
    p = np.log(np.array([0.5, 0.5]))
    q = np.log(np.array([0.25, 0.75]))
    expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
    assert kl_divergence(p, q) == pytest.approx(expected)
    assert kl_divergence(p, p) == 0


def _windows(backend):
    text = "The quick brown fox jumps over the lazy dog. " * 10
    return make_windows(backend.encode(text), window=32, max_windows=3)


def test_backend_against_itself(tiny_backend):
    result = evaluate_quality(tiny_backend, _windows(tiny_backend), reference=tiny_backend)
    assert result.top1_agreement == 1.0
    assert result.mean_kl == pytest.approx(0, abs=1e-12)
    assert result.perplexity > 1


def test_cache_setting_does_not_change_quality(tiny_backend, tiny_nocache_backend):
    windows = _windows(tiny_backend)
    a = evaluate_quality(tiny_backend, windows)
    b = evaluate_quality(tiny_nocache_backend, windows)
    assert a.perplexity == b.perplexity


def test_kl_positive_between_different_models(tiny_backend, tiny_model_factory, tiny_tokenizer):
    other = TorchBackend(
        "tiny-seed1", rung="other", model=tiny_model_factory(1), tokenizer=tiny_tokenizer
    )
    result = evaluate_quality(other, _windows(tiny_backend), reference=tiny_backend)
    assert result.mean_kl > 0
    assert result.top1_agreement < 1.0


def test_load_text_file(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("héllo\nworld", encoding="utf-8")
    assert load_text_file(path) == "héllo\nworld"
