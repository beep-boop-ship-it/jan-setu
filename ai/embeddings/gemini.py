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


class GeminiEmbeddingBackend:
    """Live Gemini text-embedding backend for JanSetu semantic retrieval."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-embedding-2",
        dimensions: int = 768,
        retry: RetryConfig | None = None,
    ) -> None:
        key = (
            api_key
            or os.getenv("JANSETU_EMBEDDING_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if not key or not key.strip():
            raise ProviderConfigurationError(
                "An API key is required for the Gemini embedding backend",
                provider="gemini-embeddings",
            )

        if not isinstance(model, str) or not model.strip():
            raise ProviderConfigurationError(
                "A Gemini embedding model must be configured",
                provider="gemini-embeddings",
            )

        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise TypeError("dimensions must be an integer")

        if not 8 <= dimensions <= 65_536:
            raise ValueError("dimensions must be within 8..65536")

        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:
            raise ProviderConfigurationError(
                "Install the 'google-genai' optional dependency",
                provider="gemini-embeddings",
            ) from exc

        self._errors = errors
        self._types = types
        self._client = genai.Client(api_key=key.strip())
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

        config = self._types.EmbedContentConfig(
            output_dimensionality=self._dimensions,
        )

        results: list[Sequence[float]] = []

        for text in values:

            async def operation(_remaining: float, text: str = text) -> Any:
                return await self._client.aio.models.embed_content(
                    model=self._model,
                    contents=text,
                    config=config,
                )

            response = await run_provider_request(
                operation,
                provider_name="gemini-embeddings",
                retry=self._retry,
                timeout_seconds=None,
                normalize_error=self._normalize_error,
            )

            embeddings = getattr(response, "embeddings", None) or []

            if len(embeddings) != 1:
                raise ProviderError(
                    "Gemini embedding request returned an unexpected "
                    f"number of embeddings: {len(embeddings)}",
                    provider="gemini-embeddings",
                )

            vector = list(getattr(embeddings[0], "values", None) or [])

            if not vector:
                raise ProviderError(
                    "Gemini embedding request returned an empty vector",
                    provider="gemini-embeddings",
                )

            results.append(vector)

        return results

    def _normalize_error(self, exc: Exception) -> ProviderError:
        api_error = getattr(self._errors, "APIError", None)

        if api_error is not None and isinstance(exc, api_error):
            code = getattr(exc, "code", None)

            if code == 429:
                return ProviderRateLimitError(
                    "Gemini embedding rate limit was exceeded",
                    provider="gemini-embeddings",
                )

            if code in {408, 504}:
                return ProviderTimeoutError(
                    "Gemini embedding request timed out",
                    provider="gemini-embeddings",
                )

            if code == 409 or (
                isinstance(code, int) and code >= 500
            ):
                return ProviderUnavailableError(
                    "Gemini embeddings are temporarily unavailable",
                    provider="gemini-embeddings",
                )

            if code in {401, 403}:
                return ProviderConfigurationError(
                    "Gemini embedding authentication or permission "
                    "configuration is invalid",
                    provider="gemini-embeddings",
                )

            return ProviderError(
                "Gemini rejected the embedding request",
                provider="gemini-embeddings",
            )

        if isinstance(exc, (ConnectionError, OSError)):
            return ProviderUnavailableError(
                "Gemini embeddings could not be reached",
                provider="gemini-embeddings",
            )

        return ProviderError(
            "Gemini embedding request failed",
            provider="gemini-embeddings",
        )