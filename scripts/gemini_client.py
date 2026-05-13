from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request

import tiktoken
from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_XAI_MODEL = "grok-4.20-non-reasoning"
XAI_BASE_URL = "https://api.x.ai/v1"
PROVIDERS = {"gemini", "xai"}

# Models tried in order when daily quota is exhausted on the primary model.
# Each has its own 20 RPD free-tier bucket, so rotating lets us run up to
# len(MODEL_ROTATION) × 20 calls per day.
MODEL_ROTATION = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite-preview",
]


def _is_daily_quota_error(exc: Exception) -> bool:
    """Return True when the error is a per-day quota exhaustion (not per-minute)."""
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg and "PerDay" in msg


@dataclass(frozen=True)
class GeminiResult:
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


def load_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to .env.")
    return genai.Client(api_key=api_key)


def selected_provider(model: str, provider: str | None = None) -> str:
    if provider:
        return provider.lower()
    env_provider = os.getenv("LLM_PROVIDER")
    if env_provider:
        return env_provider.lower()
    if model.startswith("grok-"):
        return "xai"
    return "gemini"


def provider_model(provider: str, requested_model: str) -> str:
    if provider == "xai" and requested_model == DEFAULT_MODEL:
        return os.getenv("XAI_MODEL", DEFAULT_XAI_MODEL)
    if provider == "gemini" and requested_model.startswith("grok-"):
        return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    return requested_model


def fallback_provider(primary: str) -> str | None:
    fallback = os.getenv("LLM_FALLBACK_PROVIDER", "").strip().lower()
    if not fallback:
        return None
    if fallback not in PROVIDERS:
        raise ValueError(f"Unsupported LLM fallback provider: {fallback}")
    if fallback == primary:
        return None
    return fallback


def fallback_token_count(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def usage_value(usage: Any, *names: str) -> int | None:
    if usage is None:
        return None
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None


def generate_text(
    prompt: str,
    system_instruction: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_attempts: int = 3,
    provider: str | None = None,
) -> GeminiResult:
    load_dotenv()
    llm_provider = selected_provider(model=model, provider=provider)
    if llm_provider not in PROVIDERS:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    # Build a deduplicated rotation: requested model first, then the rest.
    rotation = [model] + [m for m in MODEL_ROTATION if m != model]

    last_exc: Exception | None = None
    for candidate in rotation:
        try:
            return generate_text_once(
                prompt=prompt,
                system_instruction=system_instruction,
                provider=llm_provider,
                model=provider_model(llm_provider, candidate),
                temperature=temperature,
                max_attempts=max_attempts,
            )
        except Exception as exc:
            if _is_daily_quota_error(exc):
                last_exc = exc
                # Try the next model in the rotation.
                continue
            # Non-quota error: try the provider fallback then re-raise.
            fallback = fallback_provider(llm_provider)
            if fallback is None:
                raise
            return generate_text_once(
                prompt=prompt,
                system_instruction=system_instruction,
                provider=fallback,
                model=provider_model(fallback, candidate),
                temperature=temperature,
                max_attempts=max_attempts,
            )

    # All rotation candidates exhausted — last resort: provider fallback.
    fallback = fallback_provider(llm_provider)
    if fallback is not None:
        return generate_text_once(
            prompt=prompt,
            system_instruction=system_instruction,
            provider=fallback,
            model=provider_model(fallback, model),
            temperature=temperature,
            max_attempts=max_attempts,
        )
    raise last_exc or RuntimeError("All Gemini model rotation candidates exhausted.")


def generate_text_once(
    prompt: str,
    system_instruction: str,
    provider: str,
    model: str,
    temperature: float,
    max_attempts: int,
) -> GeminiResult:
    if provider == "xai":
        return generate_xai_text(
            prompt=prompt,
            system_instruction=system_instruction,
            model=model,
            temperature=temperature,
            max_attempts=max_attempts,
        )

    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            client = load_client()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction,
                ),
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            time.sleep(1.5 * attempt)
    else:
        raise RuntimeError("Gemini request failed") from last_error

    latency_ms = (time.perf_counter() - started) * 1000

    answer = (response.text or "").strip()
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = usage_value(usage, "prompt_token_count") or fallback_token_count(
        f"{system_instruction}\n\n{prompt}"
    )
    completion_tokens = usage_value(usage, "candidates_token_count") or fallback_token_count(answer)
    total_tokens = usage_value(usage, "total_token_count") or (prompt_tokens + completion_tokens)

    return GeminiResult(
        answer=answer,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
    )


def generate_xai_text(
    prompt: str,
    system_instruction: str,
    model: str = DEFAULT_XAI_MODEL,
    temperature: float = 0.0,
    max_attempts: int = 3,
) -> GeminiResult:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY is missing. Add it to .env.")

    started = time.perf_counter()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        req = request.Request(
            f"{os.getenv('XAI_BASE_URL', XAI_BASE_URL).rstrip('/')}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=3600) as response:
                response_json = json.loads(response.read().decode("utf-8"))
            break
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"xAI request failed: HTTP {exc.code}: {detail}")
            if attempt == max_attempts:
                raise last_error
            time.sleep(1.5 * attempt)
        except error.URLError as exc:
            last_error = RuntimeError(f"xAI request failed: {exc.reason}")
            if attempt == max_attempts:
                raise last_error
            time.sleep(1.5 * attempt)
    else:
        raise RuntimeError("xAI request failed") from last_error

    latency_ms = (time.perf_counter() - started) * 1000
    choices = response_json.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    answer = str(message.get("content") or "").strip()
    usage = response_json.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or fallback_token_count(f"{system_instruction}\n\n{prompt}")
    completion_tokens = usage.get("completion_tokens") or fallback_token_count(answer)
    total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

    return GeminiResult(
        answer=answer,
        model=str(response_json.get("model") or model),
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        total_tokens=int(total_tokens),
        latency_ms=latency_ms,
    )
