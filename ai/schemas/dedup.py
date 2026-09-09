from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


class ProblemRelationship(StrEnum):
    DUPLICATE = "duplicate"
    RELATED = "related"
    INDEPENDENT = "independent"


class DecisionSource(StrEnum):
    RULES = "rules"
    RULES_FALLBACK = "rules_fallback"
    LLM = "llm"


def _clean_required_text(value: str, *, field_name: str) -> str:
    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def _clean_unique_strings(values: list[str], *, max_item_length: int = 120) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            continue
        if len(cleaned) > max_item_length:
            raise ValueError(f"List items must not exceed {max_item_length} characters")
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


class IncomingProblem(BaseModel):
    """Minimal, bounded problem representation used by deduplication."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, max_length=128)
    title: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1, max_length=10_000)
    domain: str | None = Field(default=None, max_length=120)
    tags: list[str] = Field(default_factory=list, max_length=30)
    locality: str | None = Field(default=None, max_length=200)
    reported_at: datetime | None = None

    @field_validator("id", "domain", "locality", mode="after")
    @classmethod
    def clean_optional_fields(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)

    @field_validator("title", mode="after")
    @classmethod
    def clean_title(cls, value: str) -> str:
        return _clean_required_text(value, field_name="title")

    @field_validator("description", mode="after")
    @classmethod
    def clean_description(cls, value: str) -> str:
        return _clean_required_text(value, field_name="description")

    @field_validator("tags", mode="after")
    @classmethod
    def clean_tags(cls, values: list[str]) -> list[str]:
        return _clean_unique_strings(values)

    def embedding_text(self) -> str:
        details = [
            self.title,
            self.description,
            self.domain or "",
            " ".join(self.tags),
            self.locality or "",
        ]
        return "\n".join(value for value in details if value)


class RetrievedProblem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: IncomingProblem
    vector_similarity: float = Field(ge=-1, le=1)


class SimilaritySignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector_similarity: float = Field(ge=-1, le=1)
    title_similarity: float = Field(ge=0, le=1)
    tag_overlap: float = Field(ge=0, le=1)
    same_domain: bool | None = None
    same_locality: bool | None = None
    temporal_distance_days: int | None = Field(default=None, ge=0)
    combined_signal: float = Field(ge=0, le=1)
    contradictory_signals: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("contradictory_signals", mode="after")
    @classmethod
    def clean_contradictions(cls, values: list[str]) -> list[str]:
        return _clean_unique_strings(values, max_item_length=80)


class RelationshipDecision(BaseModel):
    """Semantic relationship plus the policy needed to act on it safely."""

    model_config = ConfigDict(extra="forbid")

    relationship: ProblemRelationship
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=500)
    decision_source: DecisionSource = DecisionSource.RULES
    provider_name: str | None = Field(default=None, max_length=100)
    requires_human_review: bool = False

    @field_validator("rationale", mode="after")
    @classmethod
    def clean_rationale(cls, value: str) -> str:
        return _clean_required_text(value, field_name="rationale")

    @field_validator("provider_name", mode="after")
    @classmethod
    def clean_provider_name(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)

    @model_validator(mode="after")
    def validate_source_metadata(self) -> "RelationshipDecision":
        if self.decision_source is DecisionSource.LLM and not self.provider_name:
            raise ValueError("LLM decisions must identify the provider")
        return self

    @computed_field
    @property
    def auto_cluster_eligible(self) -> bool:
        return (
            self.relationship is ProblemRelationship.DUPLICATE
            and self.decision_source is DecisionSource.RULES
            and not self.requires_human_review
        )


class DedupMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: IncomingProblem
    signals: SimilaritySignals
    classification: RelationshipDecision


class DedupResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incoming_problem: IncomingProblem
    matches: list[DedupMatch] = Field(default_factory=list, max_length=50)
    candidates_considered: int = Field(default=0, ge=0)
    llm_configured: bool = False
    llm_used: bool = False
    needs_human_review: bool = False

    @model_validator(mode="after")
    def validate_result(self) -> "DedupResult":
        if self.candidates_considered < len(self.matches):
            raise ValueError("candidates_considered cannot be smaller than matches")
        expected_review = any(
            item.classification.requires_human_review for item in self.matches
        )
        if self.needs_human_review != expected_review:
            raise ValueError(
                "needs_human_review must reflect the individual match decisions"
            )
        return self

    @computed_field
    @property
    def llm_available(self) -> bool:
        """Backward-compatible alias: means an LLM provider is configured."""
        return self.llm_configured

    @property
    def duplicate_candidates(self) -> list[DedupMatch]:
        return [
            item
            for item in self.matches
            if item.classification.relationship is ProblemRelationship.DUPLICATE
        ]

    @property
    def auto_cluster_candidates(self) -> list[DedupMatch]:
        return [
            item
            for item in self.matches
            if item.classification.auto_cluster_eligible
        ]

    @property
    def likely_duplicates(self) -> list[DedupMatch]:
        """Backward-compatible safe alias for automatically actionable duplicates."""
        return self.auto_cluster_candidates
