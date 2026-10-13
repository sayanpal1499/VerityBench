"""LLM client with SHA-256 disk caching, adaptive token/header rate pacing, and retry backoff."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Optional

from dotenv import load_dotenv
import httpx

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - allows offline test collection if google-genai is absent
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

# Load .env from repository root if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "llm_responses"

# Default fixed models approved after RUN-00-PREFLIGHT (PLAN.md §3.1)
DEFAULT_SUT_MODEL = "groq/openai/gpt-oss-20b"
DEFAULT_JUDGE_MODEL = "gemini/gemini-3.5-flash-lite"

# Verified rate limits with a 15% safety margin (8,000 TPM -> 6,800 TPM on Groq)
SAFE_RATE_LIMITS: dict[str, dict[str, int]] = {
    "groq": {"rpm_safe": 25, "tpm_safe": 6800},
    "gemini": {"rpm_safe": 12, "tpm_safe": 212500},
}


class LLMClientError(Exception):
    """Raised when an LLM API call fails after all retries."""

    def __init__(
        self,
        message: str,
        retry_after_sec: Optional[float] = None,
        rate_headers: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_sec = retry_after_sec
        self.rate_headers = rate_headers or {}


class _AdaptiveRateLimiter:
    """Tracks rolling 60-second request/token windows and live provider rate-limit headers."""

    def __init__(self) -> None:
        self.history: dict[str, list[tuple[float, int]]] = {"groq": [], "gemini": []}
        self.latest_headers: dict[str, dict[str, Any]] = {"groq": {}, "gemini": {}}

    def _prune(self, provider: str, now: float) -> None:
        window = self.history.get(provider, [])
        self.history[provider] = [(ts, tok) for ts, tok in window if now - ts < 60.0]

    def wait_if_needed(self, provider: str, estimated_tokens: int) -> float:
        """Sleep only as long as required by rolling 60s window or provider headers."""
        limits = SAFE_RATE_LIMITS.get(provider, {"rpm_safe": 15, "tpm_safe": 5000})
        total_slept = 0.0

        # 1. Check live provider header feedback first
        headers = self.latest_headers.get(provider, {})
        retry_after_sec = headers.get("retry_after_sec")
        rem_tokens = headers.get("remaining_tokens")
        reset_tokens_sec = headers.get("reset_tokens_sec", 0.0)
        rem_reqs = headers.get("remaining_requests")
        reset_reqs_sec = headers.get("reset_requests_sec", 0.0)

        # Respect explicit retry-after if set
        if retry_after_sec is not None and retry_after_sec > 0:
            sleep_s = min(retry_after_sec + random.uniform(0.15, 0.35), 65.0)
            time.sleep(sleep_s)
            total_slept += sleep_s
            self.latest_headers[provider]["retry_after_sec"] = None
            self.history[provider].clear()

        if rem_tokens is not None and rem_tokens < int(estimated_tokens * 1.15):
            sleep_s = max(0.5, min(reset_tokens_sec, 65.0)) + random.uniform(0.15, 0.35)
            time.sleep(sleep_s)
            total_slept += sleep_s
            self.latest_headers[provider]["remaining_tokens"] = None
            self.latest_headers[provider]["reset_tokens_sec"] = None
            self.history[provider].clear()

        if rem_reqs is not None and rem_reqs <= 1:
            sleep_s = max(0.5, min(reset_reqs_sec, 65.0)) + random.uniform(0.15, 0.35)
            time.sleep(sleep_s)
            total_slept += sleep_s
            self.latest_headers[provider]["remaining_requests"] = None
            self.latest_headers[provider]["reset_requests_sec"] = None

        # 2. Check rolling 60-second sliding window
        while True:
            now = time.monotonic()
            self._prune(provider, now)
            window = self.history.get(provider, [])
            req_count = len(window)
            tok_sum = sum(tok for _, tok in window)

            effective_est = min(estimated_tokens, limits["tpm_safe"])
            if req_count < limits["rpm_safe"] and (tok_sum + effective_est) <= limits["tpm_safe"]:
                break

            oldest_ts = window[0][0] if window else now
            wait_s = max(0.25, (60.0 - (now - oldest_ts)) + 0.25)
            time.sleep(wait_s)
            total_slept += wait_s

        return total_slept

    def record_call(
        self, provider: str, actual_tokens: int, raw_headers: Optional[dict[str, str]] = None
    ) -> None:
        self.history.setdefault(provider, []).append((time.monotonic(), max(1, actual_tokens)))
        if raw_headers:
            parsed: dict[str, Any] = {}
            lower_map = {k.lower(): v for k, v in raw_headers.items()}
            if "retry-after" in lower_map:
                try:
                    parsed["retry_after_sec"] = float(lower_map["retry-after"])
                except ValueError:
                    parsed["retry_after_sec"] = _parse_duration_seconds(lower_map["retry-after"])
            if "x-ratelimit-remaining-tokens" in lower_map:
                try:
                    parsed["remaining_tokens"] = int(lower_map["x-ratelimit-remaining-tokens"])
                except ValueError:
                    pass
            if "x-ratelimit-remaining-requests" in lower_map:
                try:
                    parsed["remaining_requests"] = int(lower_map["x-ratelimit-remaining-requests"])
                except ValueError:
                    pass
            if "x-ratelimit-reset-tokens" in lower_map:
                parsed["reset_tokens_sec"] = _parse_duration_seconds(
                    lower_map["x-ratelimit-reset-tokens"]
                )
            if "x-ratelimit-reset-requests" in lower_map:
                parsed["reset_requests_sec"] = _parse_duration_seconds(
                    lower_map["x-ratelimit-reset-requests"]
                )
            for key in (
                "x-ratelimit-limit-requests",
                "x-ratelimit-limit-tokens",
                "x-ratelimit-remaining-requests",
                "x-ratelimit-remaining-tokens",
                "x-ratelimit-reset-requests",
                "x-ratelimit-reset-tokens",
            ):
                if key in lower_map:
                    parsed[key] = lower_map[key]
            self.latest_headers[provider] = parsed


_RATE_LIMITER = _AdaptiveRateLimiter()

# Session counters for run metadata reporting
_SESSION_STATS: dict[str, int] = {
    "cache_hits": 0,
    "api_calls": 0,
    "retries": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}


def get_session_stats() -> dict[str, int]:
    """Return a copy of the cumulative session statistics."""
    return dict(_SESSION_STATS)


def reset_session_stats() -> None:
    """Reset cumulative session statistics to zero."""
    for key in _SESSION_STATS:
        _SESSION_STATS[key] = 0


def _parse_duration_seconds(value: str) -> float:
    """Parse Groq duration strings like '14.2s', '1m20s', or '450ms' into seconds."""
    if not value:
        return 0.0
    text = value.strip().lower()
    try:
        return float(text)
    except ValueError:
        pass
    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", text):
        num = float(amount)
        if unit == "h":
            total += num * 3600.0
        elif unit == "m":
            total += num * 60.0
        elif unit == "s":
            total += num
        elif unit == "ms":
            total += num / 1000.0
    return total


def _extract_retry_after_seconds(text: str) -> Optional[float]:
    """Extract explicit wait duration from rate-limit error messages if present."""
    if not text:
        return None
    # Matches e.g. "try again in 11.685s", "retry after 12s", "wait 500ms", "retry in 1m30s"
    m = re.search(
        r"(?:try again in|retry after|wait|retry in)\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s)?\s*)+)",
        text,
        re.IGNORECASE,
    )
    if m:
        val = _parse_duration_seconds(m.group(1).strip())
        return val if val > 0 else None
    return None


def _sanitize_secrets(text: str) -> str:
    """Ensure API keys never appear in error messages or logs."""
    if not text:
        return text
    sanitized = text
    for env_name in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.environ.get(env_name)
        if secret and len(secret) > 4:
            sanitized = sanitized.replace(secret, f"[REDACTED_{env_name}]")
    return sanitized


def parse_provider_and_model(model: str) -> tuple[str, str]:
    """Split 'provider/model_id' or infer provider from model prefix."""
    cleaned = model.strip()
    if "/" in cleaned:
        first, rest = cleaned.split("/", 1)
        if first.lower() in ("groq", "gemini"):
            return first.lower(), rest
    if cleaned.lower().startswith("gemini"):
        return "gemini", cleaned
    return "groq", cleaned


def compute_cache_key(
    provider: str,
    model_id: str,
    prompt_version: str,
    system_prompt: str,
    prompt: str,
    temperature: float,
    seed: Optional[int],
    json_mode: bool,
    schema_dict: Optional[dict[str, Any]],
) -> str:
    """Compute a deterministic SHA-256 hex key over all request-defining fields."""
    canonical = {
        "provider": provider,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "temperature": round(float(temperature), 6),
        "seed": seed,
        "json_mode": bool(json_mode),
        "schema": schema_dict or None,
    }
    serialized = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def call_llm(
    prompt: str,
    model: str = DEFAULT_SUT_MODEL,
    seed: Optional[int] = 42,
    temperature: float = 0.0,
    system_prompt: str = "",
    prompt_version: str = "v1",
    json_mode: bool = True,
    response_schema: Optional[dict[str, Any]] = None,
    max_output_tokens: int = 512,
    bypass_cache: bool = False,
    cache_dir: Optional[Path] = None,
    max_retries: int = 4,
) -> dict[str, Any]:
    """Call an LLM with SHA-256 disk caching, adaptive rate pacing, and retry backoff.

    Returns a dictionary containing:
      - "text": str (raw text or JSON string returned by the model)
      - "metadata": dict (provider, model_id, cache_hit, usage tokens, latency, headers, etc.)
    """
    provider, model_id = parse_provider_and_model(model)
    active_cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    active_cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = compute_cache_key(
        provider=provider,
        model_id=model_id,
        prompt_version=prompt_version,
        system_prompt=system_prompt,
        prompt=prompt,
        temperature=temperature,
        seed=seed,
        json_mode=json_mode,
        schema_dict=response_schema,
    )
    cache_path = active_cache_dir / f"{cache_key}.json"

    if not bypass_cache and cache_path.exists():
        try:
            cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_meta = dict(cached_payload.get("metadata", {}))
            cached_meta["cache_hit"] = True
            cached_meta["cache_key"] = cache_key
            _SESSION_STATS["cache_hits"] += 1
            return {"text": cached_payload["text"], "metadata": cached_meta}
        except (json.JSONDecodeError, KeyError):
            # Corrupted cache file; fall through to live call
            pass

    estimated_input_tokens = max(1, int((len(system_prompt) + len(prompt)) / 3.5))
    estimated_total_tokens = estimated_input_tokens + min(max_output_tokens, 256)
    slept_for_pacing = _RATE_LIMITER.wait_if_needed(provider, estimated_total_tokens)

    retry_count = 0
    last_error_msg = ""

    for attempt in range(max_retries + 1):
        start_t = time.perf_counter()
        try:
            if provider == "groq":
                result = _call_groq_api(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    temperature=temperature,
                    seed=seed,
                    json_mode=json_mode,
                    max_output_tokens=max_output_tokens,
                )
            elif provider == "gemini":
                result = _call_gemini_api(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    temperature=temperature,
                    seed=seed,
                    json_mode=json_mode,
                    response_schema=response_schema,
                    max_output_tokens=max_output_tokens,
                )
            else:
                raise LLMClientError(f"Unsupported provider '{provider}'.")

            latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
            in_tok = int(result.get("input_tokens", estimated_input_tokens))
            out_tok = int(result.get("output_tokens", 0))
            tot_tok = in_tok + out_tok

            _RATE_LIMITER.record_call(
                provider=provider,
                actual_tokens=tot_tok,
                raw_headers=result.get("raw_headers"),
            )

            _SESSION_STATS["api_calls"] += 1
            _SESSION_STATS["retries"] += retry_count
            _SESSION_STATS["input_tokens"] += in_tok
            _SESSION_STATS["output_tokens"] += out_tok

            metadata: dict[str, Any] = {
                "provider": provider,
                "model_id": model_id,
                "prompt_version": prompt_version,
                "cache_key": cache_key,
                "cache_hit": False,
                "bypass_cache": bypass_cache,
                "retry_count": retry_count,
                "pacing_sleep_sec": round(slept_for_pacing, 3),
                "latency_ms": latency_ms,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": tot_tok,
                "parameters_requested": {
                    "temperature": temperature,
                    "seed": seed,
                    "json_mode": json_mode,
                    "max_output_tokens": max_output_tokens,
                },
                "parameters_supported": result.get("parameters_supported", {}),
                "parameters_unsupported": result.get("parameters_unsupported", []),
                "system_fingerprint": result.get("system_fingerprint"),
                "rate_limit_headers": result.get("rate_limit_headers", {}),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

            to_save = {"text": result["text"], "metadata": metadata}
            cache_path.write_text(
                json.dumps(to_save, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return to_save

        except Exception as exc:
            last_error_msg = _sanitize_secrets(str(exc))
            is_retryable = any(
                token in last_error_msg.lower()
                for token in ("429", "rate limit", "resource_exhausted", "500", "502", "503", "504", "timeout", "connection")
            )
            if attempt < max_retries and is_retryable:
                retry_count += 1
                explicit_retry = getattr(exc, "retry_after_sec", None)
                if explicit_retry is None or explicit_retry <= 0:
                    explicit_retry = _extract_retry_after_seconds(last_error_msg)

                if explicit_retry is not None and explicit_retry > 0:
                    backoff_s = min(65.0, explicit_retry + random.uniform(0.15, 0.40))
                else:
                    backoff_s = min(60.0, (4.0 * (2.0 ** (attempt))) + random.uniform(0.5, 2.0))
                time.sleep(backoff_s)
                continue
            break

    raise LLMClientError(
        f"LLM call failed for {provider}/{model_id} after {retry_count} retries: {last_error_msg}"
    )


def _call_groq_api(
    model_id: str,
    system_prompt: str,
    prompt: str,
    temperature: float,
    seed: Optional[int],
    json_mode: bool,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Execute a chat completion request against Groq's OpenAI-compatible API."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise LLMClientError("Missing or placeholder GROQ_API_KEY in environment/.env.")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": float(temperature),
        "max_completion_tokens": int(max_output_tokens),
    }
    supported: dict[str, Any] = {"temperature": float(temperature)}
    unsupported: list[str] = []

    if seed is not None:
        payload["seed"] = int(seed)
        supported["seed"] = int(seed)

    if json_mode:
        payload["response_format"] = {"type": "json_object"}
        supported["response_format"] = "json_object"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )

    raw_headers = dict(response.headers)
    rate_headers = {
        k.lower(): v
        for k, v in raw_headers.items()
        if k.lower().startswith("x-ratelimit-") or k.lower() == "retry-after"
    }

    if response.status_code != 200:
        retry_after_val: Optional[float] = None
        if "retry-after" in rate_headers:
            try:
                retry_after_val = float(rate_headers["retry-after"])
            except ValueError:
                retry_after_val = _parse_duration_seconds(rate_headers["retry-after"])
        if retry_after_val is None or retry_after_val <= 0:
            if "x-ratelimit-reset-tokens" in rate_headers:
                retry_after_val = _parse_duration_seconds(rate_headers["x-ratelimit-reset-tokens"])
            elif "x-ratelimit-reset-requests" in rate_headers:
                retry_after_val = _parse_duration_seconds(rate_headers["x-ratelimit-reset-requests"])
        if retry_after_val is None or retry_after_val <= 0:
            retry_after_val = _extract_retry_after_seconds(response.text)

        if response.status_code == 429:
            _RATE_LIMITER.record_call("groq", 0, raw_headers=raw_headers)
        raise LLMClientError(
            f"Groq HTTP {response.status_code}: {_sanitize_secrets(response.text[:500])}",
            retry_after_sec=retry_after_val,
            rate_headers=rate_headers,
        )

    data = response.json()
    choice = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return {
        "text": choice or "",
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "system_fingerprint": data.get("system_fingerprint"),
        "parameters_supported": supported,
        "parameters_unsupported": unsupported,
        "raw_headers": raw_headers,
        "rate_limit_headers": rate_headers,
    }


def _call_gemini_api(
    model_id: str,
    system_prompt: str,
    prompt: str,
    temperature: float,
    seed: Optional[int],
    json_mode: bool,
    response_schema: Optional[dict[str, Any]],
    max_output_tokens: int,
) -> dict[str, Any]:
    """Execute a content generation request against Google Gemini API via google-genai SDK."""
    if genai is None or types is None:
        raise LLMClientError(
            "Missing 'google-genai' package required for Gemini API calls. "
            "Install project dependencies via `pip install -r requirements.txt`."
        )
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise LLMClientError("Missing or placeholder GEMINI_API_KEY in environment/.env.")

    client = genai.Client(api_key=api_key)

    config_kwargs: dict[str, Any] = {
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
    }
    supported: dict[str, Any] = {"temperature": float(temperature)}
    unsupported: list[str] = []

    if seed is not None:
        config_kwargs["seed"] = int(seed)
        supported["seed"] = int(seed)

    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
        supported["response_mime_type"] = "application/json"
        if response_schema is not None:
            config_kwargs["response_json_schema"] = response_schema
            supported["response_json_schema"] = True

    config = types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=config,
    )

    usage = getattr(response, "usage_metadata", None)
    in_tok = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    out_tok = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
    model_version = getattr(response, "model_version", model_id)

    return {
        "text": response.text or "",
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "system_fingerprint": model_version,
        "parameters_supported": supported,
        "parameters_unsupported": unsupported,
        "raw_headers": {},
        "rate_limit_headers": {},
    }
