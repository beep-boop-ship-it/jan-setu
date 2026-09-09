import pytest
from pydantic import ValidationError

from ai.orchestrator import AIOrchestrator
from ai.providers.mock import MockProvider
from ai.schemas.challenge import ChallengeAnalysis
from ai.schemas.participants import (
    ParticipantProfile,
    TeamSelection,
)


REPORT = (
    "Dirty drinking water in Kanke affects 50 residents "
    "after monsoon rain."
)


def analysis_payload() -> dict[str, object]:
    return {
        "normalized_title": "Dirty drinking water in Kanke",
        "summary": REPORT,
        "category": "Water",
        "domain": "water",
        "urgency": "high",
        "severity": "high",
        "affected_population": 50,
        "required_capabilities": [
            "water testing",
        ],
        "skills": [
            "water chemistry",
        ],
        "geographic_context": {
            "locality": "Kanke",
            "district": "Ranchi",
        },

        # Extracted facts use strings by schema contract.
        # ChallengeAnalyzer rebuilds their values from exact source quotes.
        "extracted_facts": {
            "problem": "Dirty drinking water in Kanke",
            "affected_population": (
                "affects 50 residents after monsoon rain"
            ),
        },

        "source_evidence": [
            {
                "field": "problem",
                "source_quote": (
                    "Dirty drinking water in Kanke"
                ),
            },
            {
                "field": "affected_population",
                "source_quote": (
                    "affects 50 residents after monsoon rain"
                ),
            },
        ],

        "report_quality": "complete",
        "is_sensible": True,
        "is_actionable": True,
        "quality_reason": (
            "The report identifies the civic problem, "
            "location, and affected residents."
        ),
        "confidence": "high",
    }


def participants() -> list[ParticipantProfile]:
    return [
        ParticipantProfile(
            id="student-1",
            display_name="Student One",
            role="student",
            skills=[
                "water chemistry",
            ],
            domains=[
                "Water",
            ],
            verified=True,
            available=True,
            notification_opt_in=True,
        ),
        ParticipantProfile(
            id="researcher-1",
            display_name="Researcher One",
            role="researcher",
            skills=[
                "water chemistry",
            ],
            capabilities=[
                "water testing",
            ],
            domains=[
                "Water",
            ],
            verified=True,
            available=True,
            notification_opt_in=True,
        ),
        ParticipantProfile(
            id="mentor-1",
            display_name="Mentor One",
            role="industry_mentor",
            capabilities=[
                "water testing",
            ],
            domains=[
                "Water",
            ],
            verified=True,
            available=True,
            notification_opt_in=True,
        ),
        ParticipantProfile(
            id="mentor-opted-out",
            display_name="Mentor Two",
            role="industry_mentor",
            capabilities=[
                "water testing",
            ],
            domains=[
                "Water",
            ],
            verified=True,
            available=True,
            notification_opt_in=False,
        ),
        ParticipantProfile(
            id="unavailable-student",
            display_name="Unavailable Student",
            role="student",
            skills=[
                "water chemistry",
            ],
            domains=[
                "Water",
            ],
            verified=True,
            available=False,
            notification_opt_in=True,
        ),
        ParticipantProfile(
            id="unverified-student",
            display_name="Unverified Student",
            role="student",
            skills=[
                "water chemistry",
            ],
            domains=[
                "Water",
            ],
            verified=False,
            available=True,
            notification_opt_in=True,
        ),
        ParticipantProfile(
            id="unrelated",
            display_name="Unrelated Student",
            role="student",
            skills=[
                "mobile development",
            ],
            domains=[
                "Digital Services",
            ],
            verified=True,
            available=True,
            notification_opt_in=True,
        ),
    ]


@pytest.mark.asyncio
async def test_routing_returns_only_eligible_opted_in_matches() -> None:
    orchestrator = await AIOrchestrator.build(
        provider=MockProvider(
            structured_responses={
                ChallengeAnalysis: analysis_payload(),
            }
        ),
        participants=participants(),
    )

    workflow = await orchestrator.process_citizen_report(
        REPORT,
        context={
            "district": "Ranchi",
            "locality": "Kanke",
            "pin_code": "834006",
        },
    )

    assert workflow.problem_statement is not None

    routing = await orchestrator.route_participant_invitations(
        workflow.problem_statement,
        approved=True,
    )

    assert "student-1" in routing.student_recipient_ids
    assert "researcher-1" in routing.student_recipient_ids

    assert (
        routing.mentor_recipient_ids
        == ["mentor-1"]
    )

    assert (
        "mentor-opted-out"
        not in routing.mentor_recipient_ids
    )

    assert (
        "unavailable-student"
        not in routing.student_recipient_ids
    )

    assert (
        "unverified-student"
        not in routing.student_recipient_ids
    )

    assert (
        "unrelated"
        not in routing.student_recipient_ids
    )


@pytest.mark.asyncio
async def test_routing_requires_explicit_backend_approval_flag() -> None:
    orchestrator = await AIOrchestrator.build(
        provider=MockProvider(
            structured_responses={
                ChallengeAnalysis: analysis_payload(),
            }
        ),
        participants=participants(),
    )

    workflow = await orchestrator.process_citizen_report(
        REPORT,
        context={
            "district": "Ranchi",
            "locality": "Kanke",
            "pin_code": "834006",
        },
    )

    assert workflow.problem_statement is not None

    with pytest.raises(
        PermissionError,
        match="requires an approved problem",
    ):
        await orchestrator.route_participant_invitations(
            workflow.problem_statement
        )


def test_team_selection_allows_smaller_valid_team() -> None:
    team = TeamSelection(
        mentor_id="mentor-1",
        student_ids=[
            "student-1",
            "student-2",
        ],
    )

    assert team.researcher_id is None
    assert team.student_ids == [
        "student-1",
        "student-2",
    ]


def test_team_selection_allows_four_person_team_with_researcher() -> None:
    team = TeamSelection(
        mentor_id="mentor-1",
        researcher_id="researcher-1",
        student_ids=[
            "student-1",
            "student-2",
            "student-3",
        ],
    )

    assert (
        len(team.student_ids)
        + (1 if team.researcher_id else 0)
        == 4
    )


def test_team_selection_rejects_more_than_four_participants() -> None:
    with pytest.raises(
        ValidationError,
        match="at most four participants",
    ):
        TeamSelection(
            mentor_id="mentor-1",
            researcher_id="researcher-1",
            student_ids=[
                "student-1",
                "student-2",
                "student-3",
                "student-4",
            ],
        )


def test_team_selection_rejects_person_in_multiple_positions() -> None:
    with pytest.raises(ValidationError):
        TeamSelection(
            mentor_id="mentor-1",
            researcher_id="student-1",
            student_ids=[
                "student-1",
                "student-2",
            ],
        )


def test_team_selection_rejects_mentor_as_participant() -> None:
    with pytest.raises(
        ValidationError,
        match="mentor cannot occupy",
    ):
        TeamSelection(
            mentor_id="mentor-1",
            student_ids=[
                "mentor-1",
                "student-1",
            ],
        )
