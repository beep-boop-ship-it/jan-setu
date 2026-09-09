from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pydantic.json_schema import SkipJsonSchema


class Urgency(StrEnum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Severity(StrEnum):
    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReportQuality(StrEnum):
    """AI assessment of the submitted report itself, not the workflow decision."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    GIBBERISH = "gibberish"
    UNCERTAIN = "uncertain"


class ReviewStatus(StrEnum):
    """Persisted review state. NEEDS_REVIEW is the pre-routing/legacy state."""

    NEEDS_REVIEW = "needs_review"
    AUTO_APPROVED = "auto_approved"
    CITIZEN_REVISION_REQUIRED = "citizen_revision_required"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    AUTO_REJECTED = "auto_rejected"
    HUMAN_CONFIRMED = "human_confirmed"


class ProblemCategory(StrEnum):
    """The only categories JanSetu may store for a citizen problem."""

    ROAD = "Road"
    WATER = "Water"
    ELECTRICITY = "Electricity"
    EDUCATION = "Education"
    HEALTH = "Health"
    WASTE = "Waste"
    SANITATION = "Sanitation"
    PUBLIC_SAFETY = "Public safety"
    AGRICULTURE = "Agriculture"
    EMPLOYMENT = "Employment"
    ENVIRONMENT = "Environment"
    TRANSPORTATION = "Transportation"
    PUBLIC_INFRASTRUCTURE = "Public infrastructure"
    WOMEN_AND_CHILD_WELFARE = "Women & Child welfare"
    DIGITAL_SERVICES = "Digital Services"
    OTHERS = "Others"


class GeographicContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    locality: str | None = Field(default=None, max_length=200)
    district: str | None = Field(default=None, max_length=200)
    state: str | None = Field(default=None, max_length=200)
    pin_code: str | None = Field(default=None, pattern=r"^\d{6}$")
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

    @field_validator("locality", "district", "state", "pin_code", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        cleaned = " ".join(value.split()).strip()
        return cleaned or None

class SourceEvidence(BaseModel):
    """An exact quote from the citizen report supporting one extracted fact."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1, max_length=100)
    source_quote: str = Field(min_length=1, max_length=500)

    @field_validator("field", "source_quote", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return " ".join(value.split()).strip()


class ChallengeAnalysis(BaseModel):
    """Validated AI analysis with citizen facts separated from derived metadata."""

    model_config = ConfigDict(extra="forbid")

    normalized_title: str = Field(min_length=3, max_length=180)
    summary: str = Field(min_length=3, max_length=1000)
    category: ProblemCategory = ProblemCategory.OTHERS
    domain: str | None = Field(default=None, max_length=120)
    subdomain: str | None = Field(default=None, max_length=120)
    urgency: Urgency = Urgency.UNKNOWN
    severity: Severity = Severity.UNKNOWN
    affected_population: int | None = Field(default=None, ge=0)

    required_capabilities: list[str] = Field(default_factory=list, max_length=20)
    skills: list[str] = Field(default_factory=list, max_length=20)
    constraints: list[str] = Field(default_factory=list, max_length=20)
    tags: list[str] = Field(default_factory=list, max_length=30)

    geographic_context: GeographicContext | None = None
    extracted_facts: dict[str, str] = Field(default_factory=dict, max_length=50)
    source_evidence: list[SourceEvidence] = Field(default_factory=list, max_length=50)

    unknown_fields: list[str] = Field(default_factory=list, max_length=30)
    missing_required_information: list[str] = Field(default_factory=list, max_length=20)
    fields_needing_confirmation: list[str] = Field(default_factory=list, max_length=20)

    report_quality: ReportQuality = ReportQuality.UNCERTAIN
    is_sensible: bool = False
    is_actionable: bool = False
    quality_reason: str = Field(default="", max_length=1000)
    grounding_issues: list[str] = Field(default_factory=list, max_length=30)
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    review_status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    prompt_version: str = Field(default="challenge-analysis-v3", max_length=100)
    source_report_fingerprint: SkipJsonSchema[str | None] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator(
        "required_capabilities",
        "skills",
        "constraints",
        "tags",
        "unknown_fields",
        "missing_required_information",
        "fields_needing_confirmation",
        "grounding_issues",
        mode="after",
    )
    @classmethod
    def clean_unique_strings(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = " ".join(value.split()).strip()
            if not cleaned:
                continue
            if len(cleaned) > 200:
                raise ValueError("List items must not exceed 200 characters")

            key = cleaned.casefold()
            if key not in seen:
                seen.add(key)
                result.append(cleaned)

        return result

    @field_validator("domain", "subdomain", "quality_reason", mode="before")
    @classmethod
    def normalize_optional_strings(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        return " ".join(value.split()).strip()

    @field_validator("extracted_facts", mode="after")
    @classmethod
    def validate_fact_keys(cls, values: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        seen: set[str] = set()

        for raw_key, value in values.items():
            key = " ".join(raw_key.split()).strip()
            if not key:
                raise ValueError("Extracted fact keys must not be empty")
            if len(key) > 100:
                raise ValueError("Extracted fact keys must not exceed 100 characters")

            folded = key.casefold()
            if folded in seen:
                raise ValueError(f"Duplicate extracted fact key: {key}")

            fact = " ".join(value.split()).strip()
            if not fact:
                raise ValueError("Extracted fact values must not be empty")
            if len(fact) > 500:
                raise ValueError("Extracted fact values must not exceed 500 characters")

            seen.add(folded)
            cleaned[key] = fact

        return cleaned

    @model_validator(mode="after")
    def derive_known_category_from_legacy_domain(self) -> "ChallengeAnalysis":
        """Keep older provider payloads useful while enforcing the fixed list."""

        if self.category is ProblemCategory.OTHERS and self.domain:
            domain = " ".join(self.domain.split()).casefold()
            for category in ProblemCategory:
                if domain == category.value.casefold():
                    self.category = category
                    break
        return self
