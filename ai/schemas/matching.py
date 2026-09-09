from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from ai.schemas.challenge import ConfidenceLevel, Urgency


class OrganizationType(StrEnum):
    UNIVERSITY = "university"
    INDUSTRY = "industry"
    GOVERNMENT = "government"
    NGO = "ngo"
    STARTUP = "startup"
    MSME = "msme"
    RESEARCH_LAB = "research_lab"


def _clean_text(
    value: str,
    *,
    field_name: str,
    max_length: int,
) -> str:
    cleaned = " ".join(value.split()).strip()

    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")

    if len(cleaned) > max_length:
        raise ValueError(
            f"{field_name} must not exceed {max_length} characters"
        )

    return cleaned


def _clean_terms(
    values: list[str],
    *,
    max_item_length: int = 200,
) -> list[str]:
    """
    Normalize and deduplicate matching terms.

    The default 200-character limit intentionally matches the upstream
    ChallengeAnalysis list-item contract so a valid challenge cannot fail
    merely when being converted into matching requirements.
    """

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = " ".join(value.split()).strip()

        if not cleaned:
            continue

        if len(cleaned) > max_item_length:
            raise ValueError(
                f"List items must not exceed {max_item_length} characters"
            )

        key = cleaned.casefold()

        if key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


class GeoPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    label: str | None = Field(default=None, max_length=200)

    @field_validator("label", mode="after")
    @classmethod
    def clean_label(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = " ".join(value.split()).strip()
        return cleaned or None


class ChallengeRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, max_length=128)
    title: str = Field(min_length=3, max_length=180)
    summary: str = Field(min_length=3, max_length=1000)

    # This may be broader than ChallengeAnalysis.domain, which is safe:
    # downstream schemas may accept a superset of upstream-valid values.
    domain: str | None = Field(default=None, max_length=160)

    skills: list[str] = Field(default_factory=list, max_length=50)
    required_capabilities: list[str] = Field(
        default_factory=list,
        max_length=50,
    )
    required_facilities: list[str] = Field(
        default_factory=list,
        max_length=30,
    )
    preferred_industry_capabilities: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    location: GeoPoint | None = None
    urgency: Urgency = Urgency.UNKNOWN

    constraints: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    allowed_organization_types: list[OrganizationType] = Field(
        default_factory=list,
        max_length=10,
    )

    max_distance_km: float | None = Field(
        default=None,
        gt=0,
        le=20_000,
    )

    @field_validator("id", mode="after")
    @classmethod
    def clean_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        if not cleaned:
            raise ValueError("id must not be blank")

        return cleaned

    @field_validator("title", mode="after")
    @classmethod
    def clean_title(cls, value: str) -> str:
        return _clean_text(
            value,
            field_name="title",
            max_length=180,
        )

    @field_validator("summary", mode="after")
    @classmethod
    def clean_summary(cls, value: str) -> str:
        return _clean_text(
            value,
            field_name="summary",
            max_length=1000,
        )

    @field_validator("domain", mode="after")
    @classmethod
    def clean_domain(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = " ".join(value.split()).strip()
        return cleaned or None

    @field_validator(
        "skills",
        "required_capabilities",
        "required_facilities",
        "preferred_industry_capabilities",
        "constraints",
        mode="after",
    )
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return _clean_terms(values)

    def embedding_text(self) -> str:
        values = [
            self.title,
            self.summary,
            self.domain or "",
            *self.skills,
            *self.required_capabilities,
            *self.required_facilities,
            *self.preferred_industry_capabilities,
        ]

        return "\n".join(
            value
            for value in values
            if value
        )


class PreviousProject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=180)
    summary: str = Field(default="", max_length=1000)

    domains: list[str] = Field(
        default_factory=list,
        max_length=30,
    )
    skills: list[str] = Field(
        default_factory=list,
        max_length=50,
    )

    verified: bool = False

    @field_validator("title", mode="after")
    @classmethod
    def clean_title(cls, value: str) -> str:
        return _clean_text(
            value,
            field_name="title",
            max_length=180,
        )

    @field_validator("summary", mode="after")
    @classmethod
    def clean_summary(cls, value: str) -> str:
        return " ".join(value.split()).strip()

    @field_validator("domains", "skills", mode="after")
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return _clean_terms(values)


class OrganizationCapabilityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=180)

    organization_type: OrganizationType = OrganizationType.UNIVERSITY

    description: str = Field(
        default="",
        max_length=2000,
    )

    domains: list[str] = Field(
        default_factory=list,
        max_length=40,
    )
    skills: list[str] = Field(
        default_factory=list,
        max_length=80,
    )
    research_areas: list[str] = Field(
        default_factory=list,
        max_length=60,
    )
    facilities: list[str] = Field(
        default_factory=list,
        max_length=60,
    )

    available_capacity: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    previous_projects: list[PreviousProject] = Field(
        default_factory=list,
        max_length=50,
    )

    location: GeoPoint | None = None

    industry_capabilities: list[str] = Field(
        default_factory=list,
        max_length=60,
    )

    verified: bool = False

    @field_validator("id", mode="after")
    @classmethod
    def clean_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("id must not be blank")

        return cleaned

    @field_validator("name", mode="after")
    @classmethod
    def clean_name(cls, value: str) -> str:
        return _clean_text(
            value,
            field_name="name",
            max_length=180,
        )

    @field_validator("description", mode="after")
    @classmethod
    def clean_description(cls, value: str) -> str:
        return " ".join(value.split()).strip()

    @field_validator(
        "domains",
        "skills",
        "research_areas",
        "facilities",
        "industry_capabilities",
        mode="after",
    )
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return _clean_terms(values)

    def embedding_text(self) -> str:
        projects = [
            f"{item.title} {item.summary}".strip()
            for item in self.previous_projects
            if item.verified
        ]

        values = [
            self.name,
            self.description,
            *self.domains,
            *self.skills,
            *self.research_areas,
            *self.facilities,
            *self.industry_capabilities,
            *projects,
        ]

        return "\n".join(
            value
            for value in values
            if value
        )


class RetrievedOrganization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization: OrganizationCapabilityProfile

    semantic_similarity: float = Field(
        ge=-1,
        le=1,
    )


class FeatureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=80,
    )

    value: float = Field(
        ge=0,
        le=1,
    )

    weight: float = Field(
        ge=0,
        le=1,
    )

    contribution: float = Field(
        ge=0,
        le=1,
    )

    positive_evidence: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    gaps: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    available: bool = True
    data_present: bool = True

    @field_validator(
        "positive_evidence",
        "gaps",
        mode="after",
    )
    @classmethod
    def clean_evidence(cls, values: list[str]) -> list[str]:
        # Explanatory evidence may legitimately be more verbose than
        # individual matching terms.
        return _clean_terms(
            values,
            max_item_length=300,
        )


class MatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    challenge_id: str | None = Field(
        default=None,
        max_length=128,
    )

    organization_id: str = Field(
        min_length=1,
        max_length=128,
    )

    organization_name: str = Field(
        min_length=1,
        max_length=180,
    )

    final_score: float = Field(
        ge=0,
        le=1,
    )

    semantic_score: float = Field(
        ge=0,
        le=1,
    )

    feature_breakdown: list[FeatureEvidence] = Field(
        max_length=20,
    )

    positive_evidence: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    gaps: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    hard_constraint_failures: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    explanation: str = Field(
        default="",
        max_length=1000,
    )

    confidence: ConfidenceLevel = ConfidenceLevel.LOW

    @computed_field
    @property
    def score_percent(self) -> float:
        return round(
            self.final_score * 100,
            1,
        )


class MatchingResultSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    challenge: ChallengeRequirements

    matches: list[MatchResult] = Field(
        default_factory=list,
        max_length=100,
    )

    candidates_considered: int = Field(
        default=0,
        ge=0,
    )
