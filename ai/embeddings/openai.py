from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from ai.providers.base import (
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RetryConfig,
    run_provider_request,
)


class OpenAIEmbeddingBackend:
    """Live OpenAI text-embedding backend for JanSetu semantic retrieval."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        retry: RetryConfig | None = None,
    ) -> None:
        key = (
            api_key
            or os.getenv("JANSETU_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not key or not key.strip():
            raise ProviderConfigurationError(
                "An API key is required for the OpenAI embedding backend",
                provider="openai-embeddings",
            )
        if not isinstance(model, str) or not model.strip():
            raise ProviderConfigurationError(
                "An OpenAI embedding model must be configured",
                provider="openai-embeddings",
            )
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise TypeError("dimensions must be an integer")
        if not 8 <= dimensions <= 65_536:
            raise ValueError("dimensions must be within 8..65536")

        try:
            import openai
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ProviderConfigurationError(
                "Install the 'openai' optional dependency",
                provider="openai-embeddings",
            ) from exc

        self._openai = openai
        self._client = AsyncOpenAI(api_key=key.strip(), max_retries=0)
        self._model = model.strip()
        self._dimensions = dimensions
        self._retry = retry or RetryConfig()

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_batch(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        values = list(texts)
        if not values:
            return []

        async def operation(remaining: float) -> Any:
            client = self._client.with_options(
                timeout=remaining,
                max_retries=0,
            )
            return await client.embeddings.create(
                model=self._model,
                input=values,
                dimensions=self._dimensions,
                encoding_format="float",
            )

        response = await run_provider_request(
            operation,
            provider_name="openai-embeddings",
            retry=self._retry,
            timeout_seconds=None,
            normalize_error=self._normalize_error,
        )

        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def _normalize_error(self, exc: Exception) -> ProviderError:
        openai = self._openai

        if isinstance(exc, openai.APITimeoutError):
            return ProviderTimeoutError(
                "OpenAI embedding request timed out",
                provider="openai-embeddings",
            )
        if isinstance(exc, openai.RateLimitError):
            return ProviderRateLimitError(
                "OpenAI embedding rate limit was exceeded",
                provider="openai-embeddings",
            )
        if isinstance(exc, openai.APIConnectionError):
            return ProviderUnavailableError(
                "OpenAI embeddings could not be reached",
                provider="openai-embeddings",
            )
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", None)
            if status in {408, 409} or (
                isinstance(status, int) and status >= 500
            ):
                return ProviderUnavailableError(
                    "OpenAI embeddings are temporarily unavailable",
                    provider="openai-embeddings",
                )
            if status in {401, 403}:
                return ProviderConfigurationError(
                    "OpenAI embedding authentication or permission configuration is invalid",
                    provider="openai-embeddings",
                )
            return ProviderError(
                "OpenAI rejected the embedding request",
                provider="openai-embeddings",
            )

        if isinstance(exc, (ConnectionError, OSError)):
            return ProviderUnavailableError(
                "OpenAI embeddings could not be reached",
                provider="openai-embeddings",
            )

        return ProviderError(
            "OpenAI embedding request failed",
            provider="openai-embeddings",
        )
