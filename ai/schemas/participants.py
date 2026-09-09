from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class ParticipantRole(StrEnum):
    STUDENT = "student"
    RESEARCHER = "researcher"
    INDUSTRY_MENTOR = "industry_mentor"


def _clean_identifier(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("Identifier must not be empty")

    if len(cleaned) > 128:
        raise ValueError("Identifiers must not exceed 128 characters")

    return cleaned


def _clean_terms(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = " ".join(value.split()).strip()

        if not cleaned:
            continue

        if len(cleaned) > 120:
            raise ValueError("Matching terms must not exceed 120 characters")

        key = cleaned.casefold()

        if key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


class ParticipantProfile(BaseModel):
    """Only matching-related data; contact details stay outside the AI layer."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=150)
    role: ParticipantRole

    skills: list[str] = Field(default_factory=list, max_length=50)
    domains: list[str] = Field(default_factory=list, max_length=30)
    capabilities: list[str] = Field(default_factory=list, max_length=50)

    verified: bool = False
    available: bool = False
    notification_opt_in: bool = False

    @field_validator("id", mode="after")
    @classmethod
    def clean_id(cls, value: str) -> str:
        return _clean_identifier(value)

    @field_validator("display_name", mode="after")
    @classmethod
    def clean_display_name(cls, value: str) -> str:
        cleaned = " ".join(value.split()).strip()

        if not cleaned:
            raise ValueError("display_name must not be empty")

        return cleaned

    @field_validator("skills", "domains", "capabilities", mode="after")
    @classmethod
    def clean_unique_terms(cls, values: list[str]) -> list[str]:
        return _clean_terms(values)


class ParticipantMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1, max_length=128)
    role: ParticipantRole
    score: float = Field(ge=0, le=1)

    matched_requirements: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    @field_validator("participant_id", mode="after")
    @classmethod
    def clean_participant_id(cls, value: str) -> str:
        return _clean_identifier(value)

    @field_validator("matched_requirements", mode="after")
    @classmethod
    def clean_matched_requirements(cls, values: list[str]) -> list[str]:
        return _clean_terms(values)


class ParticipantRoutingResult(BaseModel):
    """Safe recipient lists for the backend notification service."""

    model_config = ConfigDict(extra="forbid")

    student_matches: list[ParticipantMatch] = Field(
        default_factory=list,
        max_length=500,
    )
    mentor_matches: list[ParticipantMatch] = Field(
        default_factory=list,
        max_length=500,
    )

    @model_validator(mode="after")
    def validate_recipient_groups(self) -> "ParticipantRoutingResult":
        student_ids: set[str] = set()

        for match in self.student_matches:
            if match.role not in {
                ParticipantRole.STUDENT,
                ParticipantRole.RESEARCHER,
            }:
                raise ValueError(
                    "student_matches may contain only students or researchers"
                )

            if match.participant_id in student_ids:
                raise ValueError(
                    "A participant may appear only once in student_matches"
                )

            student_ids.add(match.participant_id)

        mentor_ids: set[str] = set()

        for match in self.mentor_matches:
            if match.role is not ParticipantRole.INDUSTRY_MENTOR:
                raise ValueError(
                    "mentor_matches may contain only industry mentors"
                )

            if match.participant_id in mentor_ids:
                raise ValueError(
                    "A participant may appear only once in mentor_matches"
                )

            if match.participant_id in student_ids:
                raise ValueError(
                    "A participant cannot appear in both recipient groups"
                )

            mentor_ids.add(match.participant_id)

        return self

    @computed_field
    @property
    def student_recipient_ids(self) -> list[str]:
        return [
            match.participant_id
            for match in self.student_matches
        ]

    @computed_field
    @property
    def mentor_recipient_ids(self) -> list[str]:
        return [
            match.participant_id
            for match in self.mentor_matches
        ]


class TeamSelection(BaseModel):
    """
    A mentor-selected participant team.

    A team may contain up to four participants in total.
    At most one participant may occupy the researcher position.
    The mentor is not counted as a team participant.
    """

    model_config = ConfigDict(extra="forbid")

    mentor_id: str = Field(
        min_length=1,
        max_length=128,
    )

    researcher_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )

    student_ids: list[str] = Field(
        min_length=1,
        max_length=4,
    )

    @field_validator("mentor_id", mode="after")
    @classmethod
    def clean_mentor_id(cls, value: str) -> str:
        return _clean_identifier(value)

    @field_validator("researcher_id", mode="after")
    @classmethod
    def clean_researcher_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return _clean_identifier(value)

    @field_validator("student_ids", mode="after")
    @classmethod
    def clean_student_ids(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = _clean_identifier(value)

            if cleaned in seen:
                raise ValueError(
                    "A student may appear in the team only once"
                )

            seen.add(cleaned)
            result.append(cleaned)

        return result

    @model_validator(mode="after")
    def validate_team(self) -> "TeamSelection":
        member_ids = list(self.student_ids)

        if self.researcher_id is not None:
            member_ids.append(self.researcher_id)

        if len(member_ids) > 4:
            raise ValueError(
                "The selected team may contain at most four participants"
            )

        if len(member_ids) != len(set(member_ids)):
            raise ValueError(
                "A person may occupy only one team position"
            )

        if self.mentor_id in member_ids:
            raise ValueError(
                "The mentor cannot occupy a participant team position"
            )

        return self
