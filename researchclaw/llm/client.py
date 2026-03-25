"""Lightweight OpenAI-compatible LLM client — stdlib only.

Features:
  - Model fallback chain (gpt-5.2 → gpt-5.1 → gpt-4.1 → gpt-4o)
  - Auto-detect max_tokens vs max_completion_tokens per model
  - Cloudflare User-Agent bypass
  - Exponential backoff retry with jitter
  - JSON mode support
  - Streaming disabled (sync only)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Models that require max_completion_tokens instead of max_tokens
_NEW_PARAM_MODELS = frozenset(
    {
        "o3",
        "o3-mini",
        "o4-mini",
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.4",
    }
)

_RESPONSES_MODEL_PREFIXES = (
    "gpt-",
    "o3",
    "o4",
)


def _safe_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


# Some OpenAI-compatible gateways are unstable with very large
# max_completion_tokens defaults for reasoning models.
_REASONING_MIN_COMPLETION_TOKENS = _safe_env_int(
    "RESEARCHCLAW_REASONING_MIN_TOKENS", 32768
)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class LLMResponse:
    """Parsed response from the LLM API."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    truncated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""

    base_url: str
    api_key: str
    primary_model: str = "gpt-4o"
    fallback_models: list[str] = field(
        default_factory=lambda: ["gpt-4.1", "gpt-4o-mini"]
    )
    max_tokens: int = 4096
    temperature: float = 0.7
    max_retries: int = 3
    retry_base_delay: float = 2.0
    timeout_sec: int = 300
    user_agent: str = _DEFAULT_USER_AGENT
    # MetaClaw bridge: extra headers for proxy requests
    extra_headers: dict[str, str] = field(default_factory=dict)
    # MetaClaw bridge: fallback URL if primary (proxy) is unreachable
    fallback_url: str = ""
    fallback_api_key: str = ""
    fallback_model: str = ""


class LLMClient:
    """Stateless OpenAI-compatible chat completion client."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._model_chain = [config.primary_model] + list(config.fallback_models)
        self._anthropic = None  # Will be set by from_rc_config if needed

    @classmethod
    def from_rc_config(cls, rc_config: Any) -> LLMClient:
        from researchclaw.llm import PROVIDER_PRESETS

        provider = getattr(rc_config.llm, "provider", "openai")
        preset = PROVIDER_PRESETS.get(provider, {})
        preset_base_url = preset.get("base_url")

        api_key = str(
            rc_config.llm.api_key
            or os.environ.get(rc_config.llm.api_key_env, "")
            or ""
        )
        backup_api_key = str(
            getattr(rc_config.llm, "fallback_api_key", "")
            or os.environ.get(getattr(rc_config.llm, "fallback_api_key_env", ""), "")
            or ""
        )

        # Use preset base_url if available and config doesn't override
        base_url = rc_config.llm.base_url or preset_base_url or ""
        base_urls = list(getattr(rc_config.llm, "base_urls", ()) or ())
        if not base_url and base_urls:
            base_url = base_urls[0]
            base_urls = base_urls[1:]
        else:
            base_urls = [u for u in base_urls if u and u != base_url]

        # Preserve original URL/key before MetaClaw bridge override
        # (needed for Anthropic adapter which should always talk directly
        # to the Anthropic API, not through the OpenAI-compatible proxy).
        original_base_url = base_url
        original_api_key = api_key

        # MetaClaw bridge: if enabled, point to proxy and set up fallback
        bridge = getattr(rc_config, "metaclaw_bridge", None)
        fallback_url = base_urls[0] if base_urls else ""
        fallback_api_key = backup_api_key

        if bridge and getattr(bridge, "enabled", False):
            fallback_url = base_url
            fallback_api_key = api_key
            base_url = bridge.proxy_url
            if bridge.fallback_url:
                fallback_url = bridge.fallback_url
            if bridge.fallback_api_key:
                fallback_api_key = bridge.fallback_api_key

        config = LLMConfig(
            base_url=base_url,
            api_key=api_key,
            primary_model=rc_config.llm.primary_model or "gpt-4o",
            fallback_models=list(rc_config.llm.fallback_models or []),
            timeout_sec=max(1, int(getattr(rc_config.llm, "timeout_sec", 300))),
            fallback_url=fallback_url,
            fallback_api_key=fallback_api_key or api_key,
            fallback_model=str(getattr(rc_config.llm, "fallback_model", "") or ""),
        )
        client = cls(config)

        # Detect Anthropic or Kimi-Anthropic provider — use original URL/key (not the
        # MetaClaw proxy URL which is OpenAI-compatible only).
        if provider in ("anthropic", "kimi-anthropic"):
            from .anthropic_adapter import AnthropicAdapter

            client._anthropic = AnthropicAdapter(
                original_base_url, original_api_key, config.timeout_sec
            )
        return client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        system: str | None = None,
        strip_thinking: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request with retry and fallback.

        Args:
            messages: List of {role, content} dicts.
            model: Override model (skips fallback chain).
            max_tokens: Override max token count.
            temperature: Override temperature.
            json_mode: Request JSON response format.
            system: Prepend a system message.
            strip_thinking: If True, strip <think>…</think> reasoning
                tags from the response content.  Use this when the
                output will be written to paper/script artifacts but
                NOT for general chat calls (to avoid corrupting
                legitimate content).

        Returns:
            LLMResponse with content and metadata.
        """
        if system:
            messages = [{"role": "system", "content": system}] + messages

        models = [model] if model else self._model_chain
        max_tok = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        last_error: Exception | None = None

        for m in models:
            try:
                resp = self._call_with_retry(m, messages, max_tok, temp, json_mode)
                if strip_thinking:
                    from researchclaw.utils.thinking_tags import strip_thinking_tags
                    resp = LLMResponse(
                        content=strip_thinking_tags(resp.content),
                        model=resp.model,
                        prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.completion_tokens,
                        total_tokens=resp.total_tokens,
                        finish_reason=resp.finish_reason,
                        truncated=resp.truncated,
                        raw=resp.raw,
                    )
                return resp
            except Exception as exc:  # noqa: BLE001
                logger.warning("Model %s failed: %s. Trying next.", m, exc)
                last_error = exc

        raise RuntimeError(
            f"All models failed. Last error: {last_error}"
        ) from last_error

    def preflight(self) -> tuple[bool, str]:
        """Quick connectivity check - one minimal chat call.

        Returns (success, message).
        Distinguishes: 401 (bad key), 403 (model forbidden),
                       404 (bad endpoint), 429 (rate limited), timeout.
        """
        is_reasoning = any(
            self.config.primary_model.startswith(p) for p in _NEW_PARAM_MODELS
        )
        min_tokens = 64 if is_reasoning else 1
        try:
            _ = self.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=min_tokens,
                temperature=0,
            )
            return True, f"OK - model {self.config.primary_model} responding"
        except urllib.error.HTTPError as e:
            status_map = {
                401: "Invalid API key",
                403: f"Model {self.config.primary_model} not allowed for this key",
                404: f"Endpoint not found: {self.config.base_url}",
                429: "Rate limited - try again in a moment",
            }
            msg = status_map.get(e.code, f"HTTP {e.code}")
            return False, msg
        except (urllib.error.URLError, OSError) as e:
            return False, f"Connection failed: {e}"
        except RuntimeError as e:
            # chat() wraps errors in RuntimeError; extract original HTTPError
            cause = e.__cause__
            if isinstance(cause, urllib.error.HTTPError):
                status_map = {
                    401: "Invalid API key",
                    403: f"Model {self.config.primary_model} not allowed for this key",
                    404: f"Endpoint not found: {self.config.base_url}",
                    429: "Rate limited - try again in a moment",
                }
                msg = status_map.get(cause.code, f"HTTP {cause.code}")
                return False, msg
            return False, f"All models failed: {e}"

    def _call_with_retry(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """Call with exponential backoff retry."""
        for attempt in range(self.config.max_retries):
            try:
                return self._raw_call(
                    model, messages, max_tokens, temperature, json_mode
                )
            except urllib.error.HTTPError as e:
                status = e.code
                body = ""
                try:
                    body = e.read().decode()[:500]
                except Exception:  # noqa: BLE001
                    pass

                # Non-retryable errors
                if status == 403 and "not allowed to use model" in body:
                    raise  # Model not available — let fallback handle

                # 400 is normally non-retryable, but some providers
                # (Azure OpenAI) return 400 during overload / rate-limit.
                # Retry if the body hints at a transient issue.
                if status == 400:
                    _transient_400 = any(
                        kw in body.lower()
                        for kw in ("rate limit", "ratelimit", "overloaded",
                                   "temporarily", "capacity", "throttl",
                                   "too many", "retry")
                    )
                    if not _transient_400:
                        raise  # Genuine bad request — don't retry

                # Retryable: 429 (rate limit), transient 400, 500, 502, 503, 504,
                # 529 (Anthropic overloaded)
                if status in (400, 429, 500, 502, 503, 504, 529):
                    delay = self.config.retry_base_delay * (2**attempt)
                    # Add jitter
                    import random

                    delay += random.uniform(0, delay * 0.3)
                    logger.info(
                        "Retry %d/%d for %s (HTTP %d). Waiting %.1fs.",
                        attempt + 1,
                        self.config.max_retries,
                        model,
                        status,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                raise  # Other HTTP errors
            except urllib.error.URLError:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay * (2**attempt)
                    time.sleep(delay)
                    continue
                raise

        # All retries exhausted
        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} retries for model {model}"
        )

    def _should_use_responses_api(self, model: str) -> bool:
        """Use /responses for GPT/o-series models on the primary endpoint."""
        if not any(model.startswith(prefix) for prefix in _RESPONSES_MODEL_PREFIXES):
            return False
        # Keep legacy compatibility for providers known to require chat/completions.
        base = self.config.base_url.lower()
        if "api.minimax.io" in base:
            return False
        return True

    def _messages_to_responses_input(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Convert chat-style messages to Responses API input format."""
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role", "user") or "user")
            content = str(msg.get("content", "") or "")
            converted.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                }
            )
        return converted

    def _extract_responses_content(self, data: dict[str, Any]) -> str:
        """Extract assistant text from /responses payload."""
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        parts: list[str] = []
        for item in data.get("output", []) or []:
            for block in item.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype in ("output_text", "text"):
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
        return "\n".join(parts).strip()

    def _raw_call(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """Make a single API call."""
        
        # Use Anthropic adapter if configured
        if self._anthropic:
            data = self._anthropic.chat_completion(model, messages, max_tokens, temperature, json_mode)
        else:
            # Original OpenAI logic
            # Copy messages to avoid mutating the caller's list (important for
            # retries and model-fallback — each attempt must start from the
            # original, un-modified messages).
            msgs = [dict(m) for m in messages]

            # MiniMax API requires temperature in [0, 1.0]
            _temp = temperature
            if "api.minimax.io" in self.config.base_url:
                _temp = max(0.0, min(_temp, 1.0))

            body: dict[str, Any] = {
                "model": model,
                "messages": msgs,
                "temperature": _temp,
            }

            # Use correct token parameter based on model
            if any(model.startswith(prefix) for prefix in _NEW_PARAM_MODELS):
                body["max_completion_tokens"] = max(
                    max_tokens, _REASONING_MIN_COMPLETION_TOKENS
                )
            else:
                body["max_tokens"] = max_tokens

            use_responses_api = self._should_use_responses_api(model)

            if json_mode:
                # Many OpenAI-compatible proxies serving Claude models don't
                # support the response_format parameter and return HTTP 400.
                # Fall back to a system-prompt injection for non-OpenAI models.
                if model.startswith("claude") or use_responses_api:
                    _json_hint = (
                        "You MUST respond with valid JSON only. "
                        "Do not include any text outside the JSON object."
                    )
                    # Prepend to existing system message or add as new one
                    if msgs and msgs[0]["role"] == "system":
                        msgs[0]["content"] = (
                            _json_hint + "\n\n" + msgs[0]["content"]
                        )
                    else:
                        msgs.insert(
                            0, {"role": "system", "content": _json_hint}
                        )
                else:
                    body["response_format"] = {"type": "json_object"}

            if use_responses_api:
                responses_body: dict[str, Any] = {
                    "model": model,
                    "input": self._messages_to_responses_input(msgs),
                    "temperature": _temp,
                }
                if any(model.startswith(prefix) for prefix in _NEW_PARAM_MODELS):
                    responses_body["max_output_tokens"] = max(
                        max_tokens, _REASONING_MIN_COMPLETION_TOKENS
                    )
                else:
                    responses_body["max_output_tokens"] = max_tokens
                payload = json.dumps(responses_body).encode("utf-8")
                url = f"{self.config.base_url.rstrip('/')}/responses"
                logger.debug("Using /responses for model %s on primary endpoint", model)
            else:
                payload = json.dumps(body).encode("utf-8")
                url = f"{self.config.base_url.rstrip('/')}/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.config.user_agent,
            }
            # MetaClaw bridge: inject extra headers (session ID, stage info, etc.)
            headers.update(self.config.extra_headers)

            req = urllib.request.Request(url, data=payload, headers=headers)

            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                    data = json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                # Try fallback endpoint for transient upstream failures.
                if self.config.fallback_url and exc.code in (502, 503, 504, 524, 529):
                    logger.warning(
                        "Primary endpoint HTTP %d, falling back to %s",
                        exc.code,
                        self.config.fallback_url,
                    )
                    fallback_url = (
                        f"{self.config.fallback_url.rstrip('/')}/chat/completions"
                    )
                    fallback_key = self.config.fallback_api_key or self.config.api_key
                    fallback_headers = {
                        "Authorization": f"Bearer {fallback_key}",
                        "Content-Type": "application/json",
                        "User-Agent": self.config.user_agent,
                    }
                    fallback_payload = payload
                    if self.config.fallback_model:
                        fb_body = dict(body)
                        fb_body["model"] = self.config.fallback_model
                        fb_body.pop("max_tokens", None)
                        fb_body.pop("max_completion_tokens", None)
                        if any(
                            self.config.fallback_model.startswith(prefix)
                            for prefix in _NEW_PARAM_MODELS
                        ):
                            fb_body["max_completion_tokens"] = max(
                                max_tokens, _REASONING_MIN_COMPLETION_TOKENS
                            )
                        else:
                            fb_body["max_tokens"] = max_tokens
                        fallback_payload = json.dumps(fb_body).encode("utf-8")
                    fallback_req = urllib.request.Request(
                        fallback_url, data=fallback_payload, headers=fallback_headers
                    )
                    try:
                        with urllib.request.urlopen(
                            fallback_req, timeout=self.config.timeout_sec
                        ) as resp:
                            data = json.loads(resp.read())
                    except urllib.error.HTTPError as fb_exc:
                        # If fallback key/model is unauthorized, keep primary
                        # transient error path so retries still happen on primary.
                        if fb_exc.code in (401, 403):
                            logger.warning(
                                "Fallback endpoint rejected request (HTTP %d); "
                                "continuing with primary error path.",
                                fb_exc.code,
                            )
                            raise exc
                        raise
                else:
                    raise
            except (urllib.error.URLError, OSError) as exc:
                # MetaClaw bridge / multi-base-url fallback for connection issues
                if self.config.fallback_url:
                    logger.warning(
                        "Primary endpoint unreachable, falling back to %s: %s",
                        self.config.fallback_url,
                        exc,
                    )
                    fallback_url = (
                        f"{self.config.fallback_url.rstrip('/')}/chat/completions"
                    )
                    fallback_key = self.config.fallback_api_key or self.config.api_key
                    fallback_headers = {
                        "Authorization": f"Bearer {fallback_key}",
                        "Content-Type": "application/json",
                        "User-Agent": self.config.user_agent,
                    }
                    fallback_payload = payload
                    if self.config.fallback_model:
                        fb_body = dict(body)
                        fb_body["model"] = self.config.fallback_model
                        fb_body.pop("max_tokens", None)
                        fb_body.pop("max_completion_tokens", None)
                        if any(
                            self.config.fallback_model.startswith(prefix)
                            for prefix in _NEW_PARAM_MODELS
                        ):
                            fb_body["max_completion_tokens"] = max(
                                max_tokens, _REASONING_MIN_COMPLETION_TOKENS
                            )
                        else:
                            fb_body["max_tokens"] = max_tokens
                        fallback_payload = json.dumps(fb_body).encode("utf-8")
                    fallback_req = urllib.request.Request(
                        fallback_url, data=fallback_payload, headers=fallback_headers
                    )
                    try:
                        with urllib.request.urlopen(
                            fallback_req, timeout=self.config.timeout_sec
                        ) as resp:
                            data = json.loads(resp.read())
                    except urllib.error.HTTPError as fb_exc:
                        if fb_exc.code in (401, 403):
                            logger.warning(
                                "Fallback endpoint rejected request (HTTP %d); "
                                "continuing with primary error path.",
                                fb_exc.code,
                            )
                            raise exc
                        raise
                else:
                    raise

        # Handle API error responses
        if data.get("error"):
            error_info = data["error"]
            error_msg = error_info.get("message", str(error_info))
            error_type = error_info.get("type", "api_error")
            import io
            raise urllib.error.HTTPError(
                "", 500, f"{error_type}: {error_msg}", {},
                io.BytesIO(error_msg.encode()),
            )

        usage = data.get("usage", {}) or {}

        # Chat Completions format
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            return LLMResponse(
                content=content,
                model=data.get("model", model),
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
                finish_reason=choice.get("finish_reason", ""),
                truncated=(choice.get("finish_reason", "") == "length"),
                raw=data,
            )

        # Responses API format
        if "output" in data or "output_text" in data:
            content = self._extract_responses_content(data)
            status = str(data.get("status", "") or "")
            incomplete = data.get("incomplete_details") or {}
            finish_reason = ""
            if isinstance(incomplete, dict):
                finish_reason = str(incomplete.get("reason", "") or "")
            if not finish_reason:
                finish_reason = status

            prompt_tokens = int(
                usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
            )
            completion_tokens = int(
                usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
            )
            total_tokens = int(
                usage.get("total_tokens", prompt_tokens + completion_tokens)
                or (prompt_tokens + completion_tokens)
            )

            truncated = status == "incomplete" or finish_reason in (
                "max_output_tokens",
                "length",
            )
            return LLMResponse(
                content=content,
                model=data.get("model", model),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
                truncated=truncated,
                raw=data,
            )

        raise ValueError(
            "Malformed API response: expected choices or responses output. "
            f"Got: {data}"
        )


def create_client_from_yaml(yaml_path: str | None = None) -> LLMClient:
    """Create an LLMClient from the ARC config file.

    Reads base_url and api_key from config.arc.yaml's llm section.
    """
    import yaml as _yaml

    if yaml_path is None:
        yaml_path = "config.yaml"

    with open(yaml_path, encoding="utf-8") as f:
        raw = _yaml.safe_load(f)

    llm_section = raw.get("llm", {})
    api_key = str(
        os.environ.get(
            llm_section.get("api_key_env", "OPENAI_API_KEY"),
            llm_section.get("api_key", ""),
        )
        or ""
    )

    return LLMClient(
        LLMConfig(
            base_url=llm_section.get("base_url", "https://api.openai.com/v1"),
            api_key=api_key,
            primary_model=llm_section.get("primary_model", "gpt-4o"),
            fallback_models=llm_section.get(
                "fallback_models", ["gpt-4.1", "gpt-4o-mini"]
            ),
            timeout_sec=int(llm_section.get("timeout_sec", 300)),
        )
    )
