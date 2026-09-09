from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ai.embeddings.models import Embedding, EmbeddingConfig, EmbeddingError


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_HASHING_ALIASES = {
    "contaminated": "contamination",
    "universities": "university",
    "facilities": "facility",
    "projects": "project",
}


def normalize_text(text: str) -> str:
    """Normalize searchable text without discarding Unicode letters/digits."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_TOKEN_RE.findall(normalized))


def clean_embedding_text(text: str) -> str:
    """Meaning-preserving normalization for live semantic embedding providers."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).split()).strip()


def _finite_vector(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("Vector must be a numeric sequence")

    converted: list[float] = []

    for value in values:
        if isinstance(value, bool):
            raise ValueError("Vector values must be numeric, not booleans")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Vector contains a non-numeric value") from exc

        if not math.isfinite(number):
            raise ValueError("Vector contains a non-finite value")

        converted.append(number)

    return tuple(converted)


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return bounded cosine similarity for two finite equal-sized vectors."""

    if len(left) != len(right):
        raise ValueError("Vectors must have equal dimensions")

    left_values = _finite_vector(left)
    right_values = _finite_vector(right)

    dot = sum(
        a * b
        for a, b in zip(left_values, right_values, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    similarity = dot / (left_norm * right_norm)

    if not math.isfinite(similarity):
        raise ValueError("Cosine similarity is not finite")

    return max(-1.0, min(1.0, similarity))


@runtime_checkable
class EmbeddingBackend(Protocol):
    @property
    def model(self) -> str:
        ...

    @property
    def dimensions(self) -> int:
        ...

    async def embed_batch(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        ...


class HashingEmbeddingBackend:
    """Offline feature-hashing backend for deterministic development/tests."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()

        if not isinstance(self.config, EmbeddingConfig):
            raise TypeError("config must be an EmbeddingConfig")

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    async def embed_batch(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        tokens = [
            _HASHING_ALIASES.get(token, token)
            for token in normalize_text(text).split()
        ]
        features = tokens + [
            f"{left}_{right}"
            for left, right in zip(tokens, tokens[1:])
        ]

        vector = [0.0] * self.dimensions

        for feature in features:
            digest = hashlib.blake2b(
                feature.encode("utf-8"),
                digest_size=8,
            ).digest()

            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        return vector


class EmbeddingService:
    """Validate and normalize output from a replaceable embedding backend."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        normalize_vectors: bool = True,
    ) -> None:
        if not isinstance(normalize_vectors, bool):
            raise TypeError("normalize_vectors must be a boolean")

        model = getattr(backend, "model", None)
        dimensions = getattr(backend, "dimensions", None)
        embed_batch = getattr(backend, "embed_batch", None)

        if not isinstance(model, str) or not model.strip():
            raise TypeError("Embedding backend model must be a non-empty string")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise TypeError("Embedding backend dimensions must be an integer")
        if dimensions < 1:
            raise ValueError("Embedding backend dimensions must be positive")
        if not callable(embed_batch):
            raise TypeError("Embedding backend must implement embed_batch()")

        self.backend = backend
        self.normalize_vectors = normalize_vectors
        self.model = model.strip()
        self.dimensions = dimensions
        self.normalize_backend_input = isinstance(backend, HashingEmbeddingBackend)

    async def embed(
        self,
        text: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> Embedding:
        results = await self.embed_batch(
            [text],
            metadata=[dict(metadata or {})],
        )
        return results[0]

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        metadata: Sequence[dict[str, str]] | None = None,
    ) -> list[Embedding]:
        if isinstance(texts, (str, bytes, bytearray)):
            raise TypeError("texts must be a sequence of strings")

        if metadata is not None:
            if isinstance(metadata, (str, bytes, bytearray)):
                raise TypeError("metadata must be a sequence of dictionaries")
            if len(metadata) != len(texts):
                raise ValueError("Metadata count must match text count")

        cleaned = [clean_embedding_text(text) for text in texts]
        searchable = [normalize_text(text) for text in cleaned]

        if any(not text for text in searchable):
            raise ValueError("Cannot embed empty or non-searchable text")

        if not cleaned:
            return []

        backend_inputs = searchable if self.normalize_backend_input else cleaned

        try:
            raw_vectors = list(
                await self.backend.embed_batch(backend_inputs)
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Embedding backend failed") from exc

        if len(raw_vectors) != len(cleaned):
            raise EmbeddingError(
                "Embedding backend returned the wrong number of vectors"
            )

        output: list[Embedding] = []

        for index, (raw, embedded_text) in enumerate(
            zip(raw_vectors, backend_inputs, strict=True)
        ):
            if isinstance(raw, (str, bytes, bytearray)):
                raise EmbeddingError(
                    "Embedding backend returned a non-numeric vector"
                )

            if len(raw) != self.dimensions:
                raise EmbeddingError(
                    "Embedding backend returned an unexpected dimension"
                )

            try:
                vector = _finite_vector(raw)
            except ValueError as exc:
                raise EmbeddingError(
                    "Embedding backend returned an invalid vector"
                ) from exc

            is_unit_normalized = False

            if self.normalize_vectors:
                norm = math.sqrt(
                    sum(value * value for value in vector)
                )

                if not math.isfinite(norm):
                    raise EmbeddingError(
                        "Embedding vector norm is not finite"
                    )

                if norm > 0:
                    vector = tuple(
                        value / norm
                        for value in vector
                    )
                    is_unit_normalized = True

            item_metadata = (
                dict(metadata[index])
                if metadata is not None
                else {}
            )

            try:
                embedding = Embedding(
                    vector=vector,
                    model=self.model,
                    dimensions=self.dimensions,
                    normalized_text=embedded_text,
                    text_hash=hashlib.sha256(
                        embedded_text.encode("utf-8")
                    ).hexdigest(),
                    is_unit_normalized=is_unit_normalized,
                    metadata=item_metadata,
                )
            except Exception as exc:
                raise EmbeddingError(
                    "Embedding output failed validation"
                ) from exc

            output.append(embedding)

        return output
