from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class EmbeddingError(Exception):
    """An embedding backend failed or returned invalid data."""


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for an embedding backend."""

    model: str = "jansetu-hash-embedding-v1"
    dimensions: int = 128
    normalize_vectors: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise TypeError("Embedding model must be a non-empty string")
        if len(self.model.strip()) > 200:
            raise ValueError("Embedding model name must not exceed 200 characters")

        if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int):
            raise TypeError("Embedding dimensions must be an integer")
        if not 8 <= self.dimensions <= 65_536:
            raise ValueError("Embedding dimensions must be within 8..65536")

        if not isinstance(self.normalize_vectors, bool):
            raise TypeError("normalize_vectors must be a boolean")

        object.__setattr__(self, "model", self.model.strip())


class Embedding(BaseModel):
    """Validated embedding data passed between JanSetu AI components."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    vector: tuple[float, ...] = Field(repr=False)
    model: str = Field(min_length=1, max_length=200)
    dimensions: int = Field(gt=0, le=65_536)
    normalized_text: str = Field(min_length=1, max_length=100_000, repr=False)
    text_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    is_unit_normalized: bool = False
    metadata: dict[str, str] = Field(default_factory=dict, max_length=50)

    @field_validator("model", mode="after")
    @classmethod
    def clean_model(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model must not be empty")
        return cleaned

    @field_validator("metadata", mode="after")
    @classmethod
    def validate_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}

        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not isinstance(raw_value, str):
                raise TypeError("Embedding metadata keys and values must be strings")

            key = raw_key.strip()
            item = raw_value.strip()

            if not key:
                raise ValueError("Embedding metadata keys must not be empty")
            if len(key) > 100:
                raise ValueError("Embedding metadata keys must not exceed 100 characters")
            if len(item) > 500:
                raise ValueError("Embedding metadata values must not exceed 500 characters")

            cleaned[key] = item

        return cleaned

    @model_validator(mode="after")
    def validate_vector(self) -> "Embedding":
        if len(self.vector) != self.dimensions:
            raise ValueError("Vector length does not match dimensions")

        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("Embedding vectors must contain only finite numbers")

        if self.is_unit_normalized:
            norm = math.sqrt(sum(value * value for value in self.vector))
            if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    "is_unit_normalized=True requires a unit-length vector"
                )

        return self
