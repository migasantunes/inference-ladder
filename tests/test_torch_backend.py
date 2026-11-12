import numpy as np
import pytest
import torch

from ladder.backends.torch_backend import TorchBackend

PROMPTS = [
    "The quick brown fox",
    "A language model predicts",
    "x",
    "Keys and values are stored in the cache so that later steps can reuse them.",
]
NEW_TOKENS = 24


@pytest.mark.parametrize("text", PROMPTS)
def test_cached_and_uncached_match_each_other_and_hf_generate(
    tiny_backend, tiny_nocache_backend, tiny_model_and_tokenizer, text
):
    model, _ = tiny_model_and_tokenizer
    ids = tiny_backend.encode(text)

    cached = tiny_backend.generate(ids, NEW_TOKENS).token_ids
    uncached = tiny_nocache_backend.generate(ids, NEW_TOKENS).token_ids

    prompt = torch.tensor([ids])
    with torch.inference_mode():
        reference = model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=NEW_TOKENS,
            min_new_tokens=NEW_TOKENS,  # no early stop on EOS, like our loop
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )[0, len(ids) :].tolist()

    assert cached == uncached
    assert cached == reference


@pytest.mark.parametrize("backend_fixture", ["tiny_backend", "tiny_nocache_backend"])
def test_generate_returns_exact_count_and_monotonic_times(request, backend_fixture):
    backend = request.getfixturevalue(backend_fixture)
    ids = backend.encode("The quick brown fox")
    result = backend.generate(ids, 10)

    assert len(result.token_ids) == 10
    assert len(result.token_times_s) == 10
    assert result.prompt_tokens == len(ids)
    assert result.token_times_s == sorted(result.token_times_s)  # non-decreasing
    assert result.token_times_s[0] > 0


def test_generate_rejects_empty_prompt(tiny_backend):
    with pytest.raises(ValueError):
        tiny_backend.generate([], 5)


def test_logits_shape_dtype_and_values(tiny_backend, tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    ids = np.array([tiny_backend.encode("The quick brown fox jumps")] * 2, dtype=np.int64)

    out = tiny_backend.logits(ids)

    assert out.shape == (2, ids.shape[1], len(tokenizer))
    assert out.dtype == np.float32
    with torch.inference_mode():
        direct = model(input_ids=torch.from_numpy(ids)).logits.numpy()
    np.testing.assert_array_equal(out, direct)


def test_encode_decode_round_trip(tiny_backend):
    text = "The cache stores keys."
    assert tiny_backend.decode(tiny_backend.encode(text)) == text


def test_info_for_injected_model(tiny_backend):
    info = tiny_backend.info
    assert info.engine == "pytorch"
    assert info.precision == "fp32"
    assert info.rung == "01_torch_fp32"
    assert info.size_on_disk_bytes == 0
    assert info.num_threads >= 1
    assert info.load_time_s >= 0


def test_load_from_local_directory(tiny_model_dir):
    backend = TorchBackend(str(tiny_model_dir), rung="01_torch_fp32", num_threads=1)
    weights = sum(p.stat().st_size for p in tiny_model_dir.glob("*.safetensors"))

    assert backend.info.size_on_disk_bytes == weights > 0
    assert backend.info.num_threads == 1
    assert len(backend.generate(backend.encode("The quick"), 5).token_ids) == 5
