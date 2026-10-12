"""Unit tests for src/llm_client.py covering cache keys, cache hits/bypass, errors, and secret redaction."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.llm_client import (
    LLMClientError,
    _parse_duration_seconds,
    _sanitize_secrets,
    call_llm,
    compute_cache_key,
    parse_provider_and_model,
)


def test_parse_provider_and_model() -> None:
    assert parse_provider_and_model("groq/llama-3.1-8b-instant") == ("groq", "llama-3.1-8b-instant")
    assert parse_provider_and_model("gemini/gemini-3.5-flash-lite") == (
        "gemini",
        "gemini-3.5-flash-lite",
    )
    assert parse_provider_and_model("gemini-3.8-flash") == ("gemini", "gemini-3.8-flash")
    assert parse_provider_and_model("llama-3.1-8b-instant") == ("groq", "llama-3.1-8b-instant")


def test_compute_cache_key_sensitivity() -> None:
    base_key = compute_cache_key(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        prompt_version="grounded_v1",
        system_prompt="sys",
        prompt="user",
        temperature=0.0,
        seed=42,
        json_mode=True,
        schema_dict={"type": "object"},
    )
    # Identical inputs produce identical SHA-256 hash
    same_key = compute_cache_key(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        prompt_version="grounded_v1",
        system_prompt="sys",
        prompt="user",
        temperature=0.0,
        seed=42,
        json_mode=True,
        schema_dict={"type": "object"},
    )
    assert base_key == same_key

    # Changing seed, model, prompt_version, or prompt changes the hash
    diff_seed = compute_cache_key(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        prompt_version="grounded_v1",
        system_prompt="sys",
        prompt="user",
        temperature=0.0,
        seed=43,
        json_mode=True,
        schema_dict={"type": "object"},
    )
    diff_prompt = compute_cache_key(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        prompt_version="naive_v0",
        system_prompt="sys",
        prompt="user",
        temperature=0.0,
        seed=42,
        json_mode=True,
        schema_dict={"type": "object"},
    )
    assert base_key != diff_seed
    assert base_key != diff_prompt


def test_cache_hit_and_bypass(tmp_path: Path) -> None:
    call_counter = {"count": 0}

    def fake_groq_call(**kwargs):
        call_counter["count"] += 1
        return {
            "text": f'{{"call": {call_counter["count"]}}}',
            "input_tokens": 50,
            "output_tokens": 10,
            "system_fingerprint": "fp_test",
            "parameters_supported": {"temperature": 0.0, "seed": 42},
            "parameters_unsupported": [],
            "raw_headers": {},
            "rate_limit_headers": {},
        }

    with patch("src.llm_client._call_groq_api", side_effect=fake_groq_call):
        # 1. First call misses cache and hits API
        res1 = call_llm(
            prompt="What is voltage?",
            model="groq/llama-3.1-8b-instant",
            seed=42,
            cache_dir=tmp_path,
            bypass_cache=False,
        )
        assert call_counter["count"] == 1
        assert res1["metadata"]["cache_hit"] is False
        assert res1["text"] == '{"call": 1}'

        # 2. Second identical call hits disk cache (no new provider call)
        res2 = call_llm(
            prompt="What is voltage?",
            model="groq/llama-3.1-8b-instant",
            seed=42,
            cache_dir=tmp_path,
            bypass_cache=False,
        )
        assert call_counter["count"] == 1
        assert res2["metadata"]["cache_hit"] is True
        assert res2["text"] == '{"call": 1}'

        # 3. Third call with bypass_cache=True forces a fresh API call
        res3 = call_llm(
            prompt="What is voltage?",
            model="groq/llama-3.1-8b-instant",
            seed=42,
            cache_dir=tmp_path,
            bypass_cache=True,
        )
        assert call_counter["count"] == 2
        assert res3["metadata"]["cache_hit"] is False
        assert res3["text"] == '{"call": 2}'


def test_api_error_and_secret_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_secret = "gsk_SUPER_SECRET_TEST_KEY_99999"
    monkeypatch.setenv("GROQ_API_KEY", fake_secret)

    def failing_call(**kwargs):
        raise RuntimeError(f"Auth failed for key {fake_secret} with HTTP 401")

    with patch("src.llm_client._call_groq_api", side_effect=failing_call):
        with pytest.raises(LLMClientError) as exc_info:
            call_llm(
                prompt="test",
                model="groq/llama-3.1-8b-instant",
                cache_dir=tmp_path,
                max_retries=1,
            )
        err_text = str(exc_info.value)
        assert fake_secret not in err_text
        assert "[REDACTED_GROQ_API_KEY]" in err_text


def test_parse_duration_seconds() -> None:
    assert _parse_duration_seconds("12.5s") == pytest.approx(12.5)
    assert _parse_duration_seconds("1m30s") == pytest.approx(90.0)
    assert _parse_duration_seconds("500ms") == pytest.approx(0.5)
