# TradingAgents/llm_clients/retry_utils.py
"""Retry utilities for LLM API calls with exponential backoff.

Provides a robust retry wrapper that handles:
  - API timeouts (ReadTimeout, APITimeoutError, TimeoutError)
  - Connection errors (ConnectionError, APIConnectionError)
  - Rate-limit errors (RateLimitError, 429 status)
  - Heartbeat / dropped-connection errors (IncompleteStreamError,
    PartialResponseError, and various SDK-specific network failures)

The wrapper is layered at two levels:

1. **Per-invoke retry** – each ``NormalizedChat*.invoke()`` call is
   individually wrapped so that a single LLM-tool loop step that fails can
   recover without restarting the whole graph.
2. **Graph-level retry** – ``TradingAgentsGraph._run_graph()`` retries the
   entire graph invocation (with checkpointing) for errors that escaped
   per-invoke retries, such as LangGraph orchestration failures.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_MAX_RETRIES: int = 8
DEFAULT_BASE_DELAY: float = 5.0          # seconds
DEFAULT_MAX_DELAY: float = 300.0         # 5-minute ceiling
DEFAULT_TIMEOUT: float = 600.0          # 10-minute per-request timeout

# ---------------------------------------------------------------------------
# Retryable exception types
# ---------------------------------------------------------------------------
# Collected lazily because the SDKs may not be installed.

def _get_retryable_exceptions() -> tuple[type[Exception], ...]:
    """Return the tuple of exception types that should trigger a retry."""
    exceptions: list[type[Exception]] = [
        # Python stdlib
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
        ConnectionAbortedError,
        BrokenPipeError,
        OSError,
    ]

    # openai / langchain_openai
    try:
        from openai import (
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
            APIStatusError,
        )
        exceptions += [APITimeoutError, APIConnectionError, RateLimitError]
        # 5xx server errors are transient
        _openai_api_status = APIStatusError
    except ImportError:
        pass

    # anthropic
    try:
        from anthropic import (
            APITimeoutError as AnthropicTimeout,
            APIConnectionError as AnthropicConnError,
            RateLimitError as AnthropicRateLimit,
        )
        exceptions += [AnthropicTimeout, AnthropicConnError, AnthropicRateLimit]
    except ImportError:
        pass

    # google
    try:
        from google.api_core import exceptions as google_exc
        exceptions += [
            google_exc.DeadlineExceeded,
            google_exc.ServiceUnavailable,
            google_exc.InternalServerError,
            google_exc.TooManyRequests,
            google_exc.ResourceExhausted,
            google_exc.Aborted,
        ]
    except ImportError:
        pass

    # httpx (used by openai, anthropic under the hood)
    try:
        import httpx
        exceptions += [
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
            httpx.ReadError,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ]
    except ImportError:
        pass

    # requests (used by some providers)
    try:
        import requests
        exceptions += [
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ]
    except ImportError:
        pass

    # urllib3 (bundled with requests)
    try:
        import urllib3.exceptions
        exceptions += [
            urllib3.exceptions.ReadTimeoutError,
            urllib3.exceptions.TimeoutError,
            urllib3.exceptions.ProtocolError,
            urllib3.exceptions.IncompleteRead,
        ]
    except ImportError:
        pass

    return tuple(set(exceptions))


_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] | None = None


def get_retryable_exceptions() -> tuple[type[Exception], ...]:
    """Lazily populate and cache the retryable exceptions tuple."""
    global _RETRYABLE_EXCEPTIONS
    if _RETRYABLE_EXCEPTIONS is None:
        _RETRYABLE_EXCEPTIONS = _get_retryable_exceptions()
    return _RETRYABLE_EXCEPTIONS


def is_retryable(exc: BaseException) -> bool:
    """Return True if *exc* is a transient error that warrants a retry."""
    retryable = get_retryable_exceptions()
    if isinstance(exc, retryable):
        return True

    # Heuristic checks for error messages that indicate heartbeat / stream
    # failures (some SDKs wrap them in generic RuntimeExceptions).
    msg = str(exc).lower()
    heartbeat_keywords = (
        "heartbeat",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "broken pipe",
        "incomplete",
        "partial response",
        "stream interrupted",
        "read error",
        "connection error",
        "server error",
        "503",
        "502",
        "500",
        "429",
        "rate limit",
        "capacity",
        "overloaded",
    )
    return any(kw in msg for kw in heartbeat_keywords)


F = TypeVar("F", bound=Callable[..., Any])


def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: bool = True,
    **kwargs: Any,
) -> Any:
    """Call *func* with exponential-backoff retry on transient errors.

    Args:
        func: The callable to invoke.
        *args: Positional args forwarded to *func*.
        max_retries: Maximum number of retry attempts (0 = no retry).
        base_delay: Initial delay in seconds; doubles each attempt.
        max_delay: Upper bound on the delay between retries.
        jitter: Whether to add random jitter to avoid thundering herd.
        **kwargs: Keyword args forwarded to *func*.

    Returns:
        The result of *func(*args, **kwargs)*.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc

            # Check for status-code errors that are NOT retryable (4xx except 429)
            status_code = getattr(exc, "status_code", None)
            if status_code is not None and isinstance(status_code, int):
                if 400 <= status_code < 500 and status_code != 429:
                    raise  # Client errors are not retryable

            if not is_retryable(exc):
                raise  # Non-transient error, propagate immediately

            if attempt >= max_retries:
                logger.error(
                    "LLM call failed after %d retries (attempt %d): %s",
                    max_retries,
                    attempt + 1,
                    exc,
                )
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay = random.uniform(delay * 0.5, delay * 1.0)

            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                max_retries + 1,
                delay,
                exc,
            )
            time.sleep(delay)

    # Unreachable: the loop either returns or raises
    raise last_exc  # pragma: no cover


def retry_invoke(
    bound_method: Callable[..., Any],
    input: Any,
    config: Any = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper that calls *bound_method(input, config, **kwargs)*
    with exponential-backoff retry.

    This is the primary entry point used by NormalizedChat* classes.
    *bound_method* is typically ``super().invoke`` (a bound method of the
    parent LangChain chat class).
    """
    return retry_with_backoff(
        bound_method,
        input,
        config,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        **kwargs,
    )