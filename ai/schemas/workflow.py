from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ai.schemas.challenge import (
    ChallengeAnalysis,
    GeographicContext,
    ProblemCategory,
    ReviewStatus,
    Severity,
    SourceEvidence,
    Urgency,
)


class ApprovalDecision(StrEnum):
    """Where the submitted report goes next."""

    AUTO_APPROVED = "auto_approved"
    CITIZEN_REVISION_REQUIRED = "citizen_revision_required"
    AUTO_REJECTED = "auto_rejected"
    HUMAN_REVIEW = "human_review"


# ---------------------------------------------------------------------------
# Public-output privacy helpers
# ---------------------------------------------------------------------------

_REDACTED = "[redacted]"

_EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)

# Conservative Indian/international-style phone detection.
# Word-boundary handling is deliberately avoided because numbers may start
# with '+'.
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"
)

_URL_RE = re.compile(
    r"\b(?:https?://|www\.)[^\s<>]+",
    flags=re.IGNORECASE,
)

# Common sensitive identifier formats. These are intentionally conservative;
# upstream analysis should still avoid placing PII in derived public fields.
_AADHAAR_RE = re.compile(
    r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"
)

# Keys in citizen_facts whose values must never survive verbatim into the
# public projection.
_SENSITIVE_FACT_KEY_MARKERS = {
    "name",
    "full_name",
    "citizen_name",
    "reporter_name",
    "email",
    "phone",
    "mobile",
    "contact",
    "address",
    "street",
    "house",
    "building",
    "aadhaar",
    "aadhar",
    "identity",
    "id_number",
    "identifier",
}

# Exact-location fields should not be exposed through the public projection.
# Coarse geography such as city/district/state may remain available.
_EXACT_LOCATION_KEY_MARKERS = {
    "address",
    "street",
    "house",
    "building",
    "latitude",
    "longitude",
    "lat",
    "lng",
    "lon",
    "coordinates",
    "coordinate",
    "postal_code",
    "postcode",
    "pincode",
    "pin_code",
}


def _normalise_key(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _is_sensitive_fact_key(key: str) -> bool:
    normalised = _normalise_key(key)

    if normalised in _SENSITIVE_FACT_KEY_MARKERS:
        return True

    return any(
        marker in normalised
        for marker in (
            "email",
            "phone",
            "mobile",
            "contact",
            "aadhaar",
            "aadhar",
            "address",
            "citizen_name",
            "reporter_name",
        )
    )


def _sensitive_fact_values(citizen_facts: dict[str, str]) -> list[str]:
    """
    Return private fact values that must not appear verbatim in public text.

    Only explicitly sensitive fact categories are used here. General facts
    such as city, category, population, or problem descriptions must not be
    indiscriminately removed.
    """

    result: list[str] = []

    for key, value in citizen_facts.items():
        if not _is_sensitive_fact_key(key):
            continue

        cleaned = " ".join(str(value).split()).strip()

        # Very short values are unsafe to perform substring replacement on
        # because they can accidentally redact unrelated words.
        if len(cleaned) >= 3:
            result.append(cleaned)

    # Longest-first prevents a shorter value from partially replacing a
    # longer sensitive value.
    result.sort(key=len, reverse=True)

    return result


def _redact_public_text(
    value: str,
    *,
    sensitive_values: list[str] | None = None,
) -> str:
    """Remove obvious contact/identity information from public-facing text."""

    cleaned = " ".join(value.split()).strip()

    if not cleaned:
        return cleaned

    cleaned = _EMAIL_RE.sub(_REDACTED, cleaned)
    cleaned = _URL_RE.sub(_REDACTED, cleaned)
    cleaned = _AADHAAR_RE.sub(_REDACTED, cleaned)
    cleaned = _PHONE_RE.sub(_REDACTED, cleaned)

    for sensitive_value in sensitive_values or []:
        cleaned = re.sub(
            re.escape(sensitive_value),
            _REDACTED,
            cleaned,
            flags=re.IGNORECASE,
        )

    # Collapse repeated adjacent redaction markers produced by multiple rules.
    cleaned = re.sub(
        r"(?:\[redacted\]\s*){2,}",
        f"{_REDACTED} ",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    return cleaned


def _redact_public_list(
    values: list[str],
    *,
    sensitive_values: list[str],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = _redact_public_text(
            value,
            sensitive_values=sensitive_values,
        )

        if not cleaned:
            continue

        key = cleaned.casefold()

        if key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


def _contains_exact_location_field(value: Any) -> bool:
    """
    Detect whether a geographic payload contains exact-location information.

    PublicProblemStatement is intended for broad public/student consumption.
    Exact coordinates, postal codes, street addresses, etc. are therefore
    suppressed at this projection boundary.
    """

    if not isinstance(value, dict):
        return False

    for key, child in value.items():
        normalised = _normalise_key(str(key))

        if (
            normalised in _EXACT_LOCATION_KEY_MARKERS
            and child not in (None, "", [], {})
        ):
            return True

        if isinstance(child, dict) and _contains_exact_location_field(child):
            return True

    return False


def _safe_public_location(
    location: GeographicContext | None,
) -> GeographicContext | None:
    """
    Preserve coarse geography only when the GeographicContext does not expose
    exact-location fields.

    If the schema contains populated address/coordinate-level information,
    omit location from the public DTO rather than accidentally disclosing it.
    """

    if location is None:
        return None

    payload = location.model_dump(mode="python")

    if _contains_exact_location_field(payload):
        return None

    return location.model_copy(deep=True)


class PublicProblemStatement(BaseModel):
    """
    Public/student-safe projection of an approved problem.

    Raw citizen reports, evidence, private facts, contact details, and exact
    location information must not cross this boundary.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=3, max_length=180)
    problem_summary: str = Field(min_length=3, max_length=1000)
    category: ProblemCategory
    location: GeographicContext | None = None
    urgency: Urgency
    severity: Severity

    constraints: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    unknowns: list[str] = Field(
        default_factory=list,
        max_length=30,
    )
    recommended_capabilities: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    recommended_skills: list[str] = Field(
        default_factory=list,
        max_length=20,
    )


class ProblemStatement(BaseModel):
    """
    Internal approved-problem record.

    This model may contain citizen-provided information required internally.
    Public/student APIs must use ``to_public()`` rather than serializing this
    model directly.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=3, max_length=180)
    problem_summary: str = Field(min_length=3, max_length=1000)

    citizen_report: str = Field(
        min_length=3,
        max_length=10_000,
        repr=False,
    )

    category: ProblemCategory
    location: GeographicContext | None = None
    urgency: Urgency
    severity: Severity

    citizen_facts: dict[str, str] = Field(
        default_factory=dict,
        max_length=50,
        repr=False,
    )

    source_evidence: list[SourceEvidence] = Field(
        default_factory=list,
        max_length=50,
        repr=False,
    )

    constraints: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    unknowns: list[str] = Field(
        default_factory=list,
        max_length=30,
    )
    recommended_capabilities: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    recommended_skills: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    @field_validator(
        "constraints",
        "unknowns",
        "recommended_capabilities",
        "recommended_skills",
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
                raise ValueError(
                    "List items must not exceed 200 characters"
                )

            key = cleaned.casefold()

            if key not in seen:
                seen.add(key)
                result.append(cleaned)

        return result

    def to_public(self) -> PublicProblemStatement:
        """
        Produce the privacy-safe public/student representation.

        This performs defence-in-depth redaction rather than assuming every
        upstream AI-derived field is already free of private information.
        """

        sensitive_values = _sensitive_fact_values(self.citizen_facts)

        title = _redact_public_text(
            self.title,
            sensitive_values=sensitive_values,
        )

        problem_summary = _redact_public_text(
            self.problem_summary,
            sensitive_values=sensitive_values,
        )

        # Redaction can theoretically reduce malformed input below the DTO's
        # minimum length. Fail closed rather than leaking the original value.
        if len(title) < 3:
            title = "Community problem"

        if len(problem_summary) < 3:
            problem_summary = "Further public details are unavailable."

        return PublicProblemStatement(
            title=title,
            problem_summary=problem_summary,
            category=self.category,
            location=_safe_public_location(self.location),
            urgency=self.urgency,
            severity=self.severity,
            constraints=_redact_public_list(
                self.constraints,
                sensitive_values=sensitive_values,
            ),
            unknowns=_redact_public_list(
                self.unknowns,
                sensitive_values=sensitive_values,
            ),
            recommended_capabilities=_redact_public_list(
                self.recommended_capabilities,
                sensitive_values=sensitive_values,
            ),
            recommended_skills=_redact_public_list(
                self.recommended_skills,
                sensitive_values=sensitive_values,
            ),
        )


class ReportWorkflowResult(BaseModel):
    """Complete output of the AI-owned report-intake workflow."""

    model_config = ConfigDict(extra="forbid")

    decision: ApprovalDecision

    decision_reasons: list[str] = Field(
        min_length=1,
        max_length=20,
    )

    analysis: ChallengeAnalysis
    problem_statement: ProblemStatement | None = None

    @field_validator("decision_reasons", mode="after")
    @classmethod
    def clean_decision_reasons(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = " ".join(value.split()).strip()

            if not cleaned:
                continue

            if len(cleaned) > 500:
                raise ValueError(
                    "Decision reasons must not exceed 500 characters"
                )

            key = cleaned.casefold()

            if key not in seen:
                seen.add(key)
                result.append(cleaned)

        if not result:
            raise ValueError(
                "At least one decision reason is required"
            )

        return result

    @model_validator(mode="after")
    def decision_matches_payload(self) -> "ReportWorkflowResult":
        expected_status = {
            ApprovalDecision.AUTO_APPROVED:
                ReviewStatus.AUTO_APPROVED,
            ApprovalDecision.CITIZEN_REVISION_REQUIRED:
                ReviewStatus.CITIZEN_REVISION_REQUIRED,
            ApprovalDecision.AUTO_REJECTED:
                ReviewStatus.AUTO_REJECTED,
            ApprovalDecision.HUMAN_REVIEW:
                ReviewStatus.HUMAN_REVIEW_REQUIRED,
        }[self.decision]

        if self.analysis.review_status is not expected_status:
            raise ValueError(
                "Workflow decision and analysis review_status do not match"
            )

        if self.decision is ApprovalDecision.AUTO_APPROVED:
            if self.problem_statement is None:
                raise ValueError(
                    "An auto-approved report must include a problem statement"
                )
        elif self.problem_statement is not None:
            raise ValueError(
                "Only an auto-approved report may include a problem statement"
            )

        return self
