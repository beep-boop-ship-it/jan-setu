from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ai.schemas.challenge import ConfidenceLevel


class ImageEvidenceStatus(StrEnum):
    SUPPORTS = "supports"
    PARTIALLY_SUPPORTS = "partially_supports"
    UNCLEAR = "unclear"
    UNRELATED = "unrelated"
    CONTRADICTS = "contradicts"


class ImageEvidenceAnalysis(BaseModel):
    """Structured assessment of uploaded photos against a citizen report."""

    model_config = ConfigDict(extra="forbid")

    status: ImageEvidenceStatus
    confidence: ConfidenceLevel = ConfidenceLevel.LOW

    usable_image_count: int = Field(
        ge=0,
        le=5,
    )

    relevant_image_count: int = Field(
        ge=0,
        le=5,
    )

    summary: str = Field(
        min_length=1,
        max_length=500,
    )

    concerns: list[str] = Field(
        default_factory=list,
        max_length=10,
    )

    @field_validator(
        "summary",
        mode="before",
    )
    @classmethod
    def clean_summary(
        cls,
        value: object,
    ) -> object:
        if isinstance(value, str):
            return " ".join(
                value.split()
            ).strip()

        return value

    @field_validator(
        "concerns",
        mode="after",
    )
    @classmethod
    def clean_concerns(
        cls,
        values: list[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = " ".join(
                value.split()
            ).strip()

            if not cleaned:
                continue

            if len(cleaned) > 300:
                raise ValueError(
                    "Image-evidence concerns must not exceed "
                    "300 characters"
                )

            key = cleaned.casefold()

            if key not in seen:
                seen.add(key)
                result.append(cleaned)

        return result

    @model_validator(mode="after")
    def counts_are_consistent(
        self,
    ) -> "ImageEvidenceAnalysis":
        if (
            self.relevant_image_count
            > self.usable_image_count
        ):
            raise ValueError(
                "relevant_image_count cannot exceed "
                "usable_image_count"
            )

        return self