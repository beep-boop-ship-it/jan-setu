from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from ai.providers.base import (
    ProviderError,
    ProviderResponseError,
    RetryConfig,
    StructuredT,
    validate_generation_request,
)


class MockProvider:
    """Deterministic, zero-network provider for tests and local development."""

    def __init__(
        self,
        *,
        structured_responses: Mapping[type[BaseModel], list[Any] | Any] | None = None,
        text_responses: list[str] | None = None,
        responder: Callable[[str, type[BaseModel]], Any] | None = None,
        default_timeout_seconds: float | None = None,
    ) -> None:
        self._structured: dict[type[BaseModel], deque[Any]] = defaultdict(deque)

        for model, responses in (structured_responses or {}).items():
            if not isinstance(model, type) or not issubclass(model, BaseModel):
                raise TypeError(
                    "structured_responses keys must be pydantic BaseModel subclasses"
                )

            values = responses if isinstance(responses, list) else [responses]
            self._structured[model].extend(values)

        self._text: deque[str] = deque(
            text_responses
            if text_responses is not None
            else ["Explanation unavailable; deterministic evidence is shown."]
        )
        self._responder = responder
        if default_timeout_seconds is None:
            self._default_timeout_seconds = RetryConfig().timeout_seconds
        else:
            if isinstance(default_timeout_seconds, bool) or not isinstance(
                default_timeout_seconds, (int, float)
            ):
                raise TypeError("default_timeout_seconds must be a finite number")
            self._default_timeout_seconds = float(default_timeout_seconds)

        if (
            not math.isfinite(self._default_timeout_seconds)
            or self._default_timeout_seconds <= 0
        ):
            raise ValueError("default_timeout_seconds must be finite and > 0")

        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "mock"

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

        self.calls.append(("structured", response_model.__name__))

        try:
            if self._structured[response_model]:
                payload = self._structured[response_model].popleft()
            elif self._responder is not None:
                payload = self._responder(prompt, response_model)
            else:
                raise ProviderResponseError(
                    f"No mock response configured for {response_model.__name__}",
                    provider=self.name,
                )

            if isinstance(payload, response_model):
                return payload

            return response_model.model_validate(payload)

        except ProviderError:
            # Tests may deliberately inject provider failures such as timeouts
            # or rate limits. Preserve those exact provider-domain exceptions.
            raise
        except (ValidationError, TypeError, ValueError) as exc:
            raise ProviderResponseError(
                f"Invalid mock response for {response_model.__name__}",
                provider=self.name,
            ) from exc
        except Exception as exc:
            raise ProviderResponseError(
                f"Mock responder failed for {response_model.__name__}",
                provider=self.name,
            ) from exc

    async def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
    ) -> str:
        validate_generation_request(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

        self.calls.append(("text", "str"))

        if not self._text:
            raise ProviderResponseError(
                "No mock text response configured",
                provider=self.name,
            )

        value = self._text.popleft()

        if not isinstance(value, str) or not value.strip():
            raise ProviderResponseError(
                "Mock text response must be a non-empty string",
                provider=self.name,
            )

        return value.strip()
