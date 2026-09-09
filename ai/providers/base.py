from __future__ import annotations

import asyncio
import math
import os
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from time import monotonic
from typing import Protocol, TypeVar

from pydantic import BaseModel


StructuredT = TypeVar("StructuredT", bound=BaseModel)
ResultT = TypeVar("ResultT")

MAX_PROMPT_CHARACTERS = 100_000
MAX_SYSTEM_PROMPT_CHARACTERS = 50_000


class AIError(Exception):
    """Base exception for the JanSetu AI domain."""


class AIRequestError(AIError):
    """The caller supplied an invalid AI generation request."""


class ProviderError(AIError):
    """A provider could not complete a request."""

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        self.provider = provider
        prefix = f"[{provider}] " if provider else ""
        super().__init__(prefix + message)


class ProviderConfigurationError(ProviderError):
    """Provider configuration or dependencies are invalid."""


class ProviderTransientError(ProviderError):
    """A temporary provider failure that may succeed when retried."""


class ProviderTimeoutError(ProviderTransientError):
    """A provider request exceeded its timeout budget."""


class ProviderRateLimitError(ProviderTransientError):
    """A provider rejected the request because of rate limiting."""


class ProviderUnavailableError(ProviderTransientError):
    """A provider is temporarily unavailable."""


class ProviderResponseError(ProviderError):
    """Provider content could not be parsed or validated."""


class FallbackExhaustedError(ProviderError):
    """No provider in a fallback chain completed successfully."""

    def __init__(self, failures: tuple[tuple[str, ProviderError], ...]) -> None:
        self.failures = failures
        summary = ", ".join(
            f"{provider}={type(error).__name__}"
            for provider, error in failures
        )
        message = f"Fallback exhausted after {len(failures)} failed provider attempt(s)"
        if summary:
            message += f" ({summary})"
        super().__init__(message, provider="fallback")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Expected a string or None")
    cleaned = value.strip()
    return cleaned or None


def _validated_timeout(value: float, *, name: str = "timeout_seconds") -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise AIRequestError(f"{name} must be a finite number")
    if value <= 0:
        raise AIRequestError(f"{name} must be > 0")
    return float(value)


def validate_generation_request(
    prompt: str,
    *,
    system_prompt: str | None,
    temperature: float,
    timeout_seconds: float | None,
    response_model: type[BaseModel] | None = None,
) -> None:
    """Validate a request before any provider/network call."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise AIRequestError("prompt must be a non-empty string")
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise AIRequestError(
            f"prompt exceeds {MAX_PROMPT_CHARACTERS:,} characters"
        )

    if system_prompt is not None:
        if not isinstance(system_prompt, str):
            raise AIRequestError("system_prompt must be a string or None")
        if len(system_prompt) > MAX_SYSTEM_PROMPT_CHARACTERS:
            raise AIRequestError("system_prompt is too large")

    if not _is_number(temperature) or not math.isfinite(temperature):
        raise AIRequestError("temperature must be a finite number")
    if not 0 <= temperature <= 2:
        raise AIRequestError("temperature must be within 0..2")

    if timeout_seconds is not None:
        _validated_timeout(timeout_seconds)

    if response_model is not None and (
        not isinstance(response_model, type)
        or not issubclass(response_model, BaseModel)
    ):
        raise AIRequestError(
            "response_model must be a pydantic BaseModel subclass"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RetryConfig:
    timeout_seconds: float = 20.0
    max_attempts: int = 2
    base_delay_seconds: float = 0.25
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 4.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if not _is_number(self.timeout_seconds):
            raise TypeError("timeout_seconds must be a finite number")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and > 0")
        if self.timeout_seconds > 120:
            raise ValueError("timeout_seconds must be <= 120")

        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise TypeError("max_attempts must be an integer")
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("max_attempts must be within 1..5")

        numeric_fields = (
            ("base_delay_seconds", self.base_delay_seconds, 0.0),
            ("backoff_multiplier", self.backoff_multiplier, 1.0),
            ("max_delay_seconds", self.max_delay_seconds, 0.0),
            ("jitter_ratio", self.jitter_ratio, 0.0),
        )
        for field_name, value, minimum in numeric_fields:
            if not _is_number(value):
                raise TypeError(f"{field_name} must be a finite number")
            if not math.isfinite(value) or value < minimum:
                raise ValueError(
                    f"{field_name} must be finite and >= {minimum}"
                )

        if self.jitter_ratio > 1:
            raise ValueError("jitter_ratio must be <= 1")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                "max_delay_seconds must be >= base_delay_seconds"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderConfig:
    name: str = "mock"
    model: str | None = None
    api_key: str | None = field(default=None, repr=False)
    retry: RetryConfig = field(default_factory=RetryConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ProviderConfigurationError("Provider name must be a string")

        try:
            name = _clean_optional(self.name)
            model = _clean_optional(self.model)
            api_key = _clean_optional(self.api_key)
        except TypeError as exc:
            raise ProviderConfigurationError(str(exc)) from exc

        if not name:
            raise ProviderConfigurationError("Provider name cannot be empty")
        if not isinstance(self.retry, RetryConfig):
            raise ProviderConfigurationError(
                "retry must be a RetryConfig instance"
            )

        object.__setattr__(self, "name", name.lower())
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "api_key", api_key)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "ProviderConfig":
        env = os.environ if environ is None else environ

        def parse_float(name: str, default: float) -> float:
            raw = _clean_optional(env.get(name))
            if raw is None:
                return default
            try:
                value = float(raw)
            except ValueError as exc:
                raise ProviderConfigurationError(
                    f"{name} must be a number, got {raw!r}"
                ) from exc
            if not math.isfinite(value):
                raise ProviderConfigurationError(f"{name} must be finite")
            return value

        def parse_int(name: str, default: int) -> int:
            raw = _clean_optional(env.get(name))
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ProviderConfigurationError(
                    f"{name} must be an integer, got {raw!r}"
                ) from exc

        defaults = RetryConfig()
        try:
            retry = RetryConfig(
                timeout_seconds=parse_float(
                    "JANSETU_AI_TIMEOUT_SECONDS",
                    defaults.timeout_seconds,
                ),
                max_attempts=parse_int(
                    "JANSETU_AI_MAX_ATTEMPTS",
                    defaults.max_attempts,
                ),
                base_delay_seconds=parse_float(
                    "JANSETU_AI_BASE_DELAY_SECONDS",
                    defaults.base_delay_seconds,
                ),
                backoff_multiplier=parse_float(
                    "JANSETU_AI_BACKOFF_MULTIPLIER",
                    defaults.backoff_multiplier,
                ),
                max_delay_seconds=parse_float(
                    "JANSETU_AI_MAX_DELAY_SECONDS",
                    defaults.max_delay_seconds,
                ),
                jitter_ratio=parse_float(
                    "JANSETU_AI_JITTER_RATIO",
                    defaults.jitter_ratio,
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ProviderConfigurationError(
                f"Invalid retry configuration: {exc}"
            ) from exc

        return cls(
            name=_clean_optional(env.get("JANSETU_AI_PROVIDER")) or "mock",
            model=_clean_optional(env.get("JANSETU_AI_MODEL")),
            api_key=_clean_optional(env.get("JANSETU_AI_API_KEY")),
            retry=retry,
        )


class AIProvider(Protocol):
    """Provider-neutral generation contract."""

    @property
    def name(self) -> str:
        ...

    @property
    def default_timeout_seconds(self) -> float:
        ...

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredT],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
    ) -> StructuredT:
        ...

def _retry_delay(config: RetryConfig, retry_index: int) -> float:
    base = min(
        config.max_delay_seconds,
        config.base_delay_seconds
        * (config.backoff_multiplier ** retry_index),
    )
    if base <= 0 or config.jitter_ratio <= 0:
        return base

    factor = random.uniform(
        1.0 - config.jitter_ratio,
        1.0 + config.jitter_ratio,
    )
    return min(config.max_delay_seconds, max(0.0, base * factor))


async def run_provider_request(
    operation: Callable[[float], Awaitable[ResultT]],
    *,
    provider_name: str,
    retry: RetryConfig,
    timeout_seconds: float | None,
    normalize_error: Callable[[Exception], ProviderError],
) -> ResultT:
    """Run one provider call with one deadline covering retries and backoff."""

    timeout = (
        retry.timeout_seconds
        if timeout_seconds is None
        else _validated_timeout(timeout_seconds)
    )
    deadline = monotonic() + timeout
    last_error: ProviderError | None = None

    for attempt in range(retry.max_attempts):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ProviderTimeoutError(
                "Provider request exhausted its total timeout budget",
                provider=provider_name,
            ) from last_error

        try:
            async with asyncio.timeout(remaining):
                return await operation(remaining)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            error = ProviderTimeoutError(
                "Provider request timed out",
                provider=provider_name,
            )
            error.__cause__ = exc
        except ProviderError as exc:
            error = exc
        except Exception as exc:
            error = normalize_error(exc)
            if error.__cause__ is None:
                error.__cause__ = exc

        if not isinstance(error, ProviderTransientError):
            raise error

        last_error = error
        if attempt + 1 >= retry.max_attempts:
            raise error

        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ProviderTimeoutError(
                "Provider request exhausted its total timeout budget",
                provider=provider_name,
            ) from error

        delay = _retry_delay(retry, attempt)
        if delay <= 0:
            continue
        if delay >= remaining:
            raise ProviderTimeoutError(
                "Retry delay would exceed the remaining timeout budget",
                provider=provider_name,
            ) from error

        await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


class FallbackProvider:
    """Try providers in order while preserving one end-to-end timeout budget."""

    __slots__ = ("_providers", "_default_timeout_seconds")

    def __init__(
        self,
        *providers: AIProvider,
        default_timeout_seconds: float | None = None,
    ) -> None:
        if not providers:
            raise ValueError("At least one provider is required")

        self._providers = tuple(providers)
        default_timeout = (
            providers[0].default_timeout_seconds
            if default_timeout_seconds is None
            else default_timeout_seconds
        )
        try:
            self._default_timeout_seconds = _validated_timeout(
                default_timeout,
                name="default_timeout_seconds",
            )
        except AIRequestError as exc:
            raise ValueError(str(exc)) from exc

    @property
    def providers(self) -> tuple[AIProvider, ...]:
        return self._providers

    @property
    def name(self) -> str:
        return "fallback:" + ",".join(
            provider.name for provider in self._providers
        )

    @property
    def default_timeout_seconds(self) -> float:
        return self._default_timeout_seconds

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredT],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
    ) -> StructuredT:
        validate_generation_request(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            response_model=response_model,
        )
        return await self._run_with_fallback(
            lambda provider, budget: provider.generate_structured(
                prompt,
                response_model,
                system_prompt=system_prompt,
                temperature=temperature,
                timeout_seconds=budget,
            ),
            timeout_seconds=timeout_seconds,
        )

    async def _run_with_fallback(
        self,
        operation: Callable[[AIProvider, float], Awaitable[ResultT]],
        *,
        timeout_seconds: float | None,
    ) -> ResultT:
        failures: list[tuple[str, ProviderError]] = []

        total_timeout = (
            self._default_timeout_seconds
            if timeout_seconds is None
            else _validated_timeout(timeout_seconds)
        )
        deadline = monotonic() + total_timeout

        for index, provider in enumerate(self._providers):
            remaining = deadline - monotonic()
            if remaining <= 0:
                break

            providers_left = len(self._providers) - index
            provider_budget = remaining / providers_left

            try:
                async with asyncio.timeout(provider_budget):
                    return await operation(provider, provider_budget)
            except asyncio.CancelledError:
                raise
            except TimeoutError as exc:
                error = ProviderTimeoutError(
                    "Provider exceeded its fallback-chain time allocation",
                    provider=provider.name,
                )
                error.__cause__ = exc
                failures.append((provider.name, error))
            except ProviderError as exc:
                failures.append((provider.name, exc))

        if failures:
            raise FallbackExhaustedError(
                tuple(failures)
            ) from failures[-1][1]

        raise ProviderTimeoutError(
            "Fallback timeout budget was exhausted before a provider was attempted",
            provider=self.name,
        )
