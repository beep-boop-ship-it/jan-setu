import pytest

from ai.embeddings.service import (
    EmbeddingService,
    HashingEmbeddingBackend,
)
from ai.orchestrator import (
    AIOrchestrator,
    AIStage,
)
from ai.providers.mock import MockProvider
from ai.schemas.challenge import ChallengeAnalysis
from ai.schemas.workflow import ApprovalDecision


REPORT = (
    "Dirty drinking water in Kanke affects 50 residents "
    "after monsoon rain."
)

LOCATION_CONTEXT = {
    "district": "Ranchi",
    "locality": "Kanke",
    "pin_code": "834006",
}

CONTEXT = {
    **LOCATION_CONTEXT,
    "title": "Dirty drinking water in Kanke",
}


def complete_analysis(
    **updates: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "normalized_title":
            "Dirty drinking water in Kanke",

        "summary":
            REPORT,

        "category":
            "Water",

        "domain":
            "water",

        "urgency":
            "high",

        "severity":
            "high",

        "affected_population":
            50,

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

        "extracted_facts": {
            "problem":
                "Dirty drinking water in Kanke",

            "affected_population": (
                "affects 50 residents "
                "after monsoon rain"
            ),
        },

        "source_evidence": [
            {
                "field":
                    "problem",

                "source_quote":
                    "Dirty drinking water in Kanke",
            },
            {
                "field":
                    "affected_population",

                "source_quote": (
                    "affects 50 residents "
                    "after monsoon rain"
                ),
            },
            {
                "field":
                    "locality",

                "source_quote":
                    "Kanke",
            },
        ],

        "report_quality":
            "complete",

        "is_sensible":
            True,

        "is_actionable":
            True,

        "quality_reason": (
            "The report clearly states the civic problem, "
            "location, and affected residents."
        ),

        "confidence":
            "high",
    }

    payload.update(
        updates
    )

    return payload


@pytest.mark.asyncio
async def test_complete_grounded_report_is_auto_approved_and_formatted() -> None:
    provider = MockProvider(
        structured_responses={
            ChallengeAnalysis:
                complete_analysis(),
        }
    )

    orchestrator = await AIOrchestrator.build(
        provider=provider
    )

    stages: list[AIStage] = []

    result = await orchestrator.process_citizen_report(
        REPORT,
        context=CONTEXT,
        on_stage=stages.append,
    )

    assert (
        result.decision
        is ApprovalDecision.AUTO_APPROVED
    )

    assert (
        result.analysis.review_status
        == "auto_approved"
    )

    assert (
        result.problem_statement
        is not None
    )

    assert (
        result.problem_statement.category
        == "Water"
    )

    assert (
        result.problem_statement.citizen_report
        == REPORT
    )

    # The optional population scalar is deliberately withheld.
    assert (
        result.analysis.affected_population
        is None
    )

    # The citizen-provided population claim remains preserved as evidence.
    assert (
        "affected_population"
        in result.analysis.extracted_facts
    )

    assert any(
        evidence.field
        == "affected_population"
        for evidence
        in result.analysis.source_evidence
    )

    assert (
        "problem"
        in result.analysis.extracted_facts
    )

    assert stages == [
        AIStage.UNDERSTANDING_PROBLEM,
        AIStage.VALIDATING_REPORT,
        AIStage.FORMATTING_PROBLEM_STATEMENT,
        AIStage.CHALLENGE_READY,
    ]


@pytest.mark.asyncio
async def test_hallucinated_fact_is_removed_and_sent_to_human_review() -> None:
    payload = complete_analysis(
        affected_population=999,
        extracted_facts={
            "affected_population":
                "999 residents",

            "cause":
                "factory chemical leak",
        },
        source_evidence=[
            {
                "field":
                    "cause",

                "source_quote":
                    "factory chemical leak",
            },
        ],
    )

    orchestrator = await AIOrchestrator.build(
        provider=MockProvider(
            structured_responses={
                ChallengeAnalysis:
                    payload,
            }
        )
    )

    result = await orchestrator.process_citizen_report(
        REPORT,
        context=CONTEXT,
    )

    assert (
        result.decision
        is ApprovalDecision.HUMAN_REVIEW
    )

    assert (
        result.problem_statement
        is None
    )

    assert (
        result.analysis.affected_population
        is None
    )

    assert (
        "cause"
        not in result.analysis.extracted_facts
    )

    assert (
        result.analysis.grounding_issues
    )


@pytest.mark.asyncio
async def test_irrelevant_evidence_cannot_satisfy_auto_approval_gate() -> None:
    """
    Population/location evidence alone cannot justify automatic approval.
    At least one substantive grounded civic-problem fact is required.
    """

    payload = complete_analysis(
        extracted_facts={
            "affected_population": (
                "affects 50 residents "
                "after monsoon rain"
            ),
        },
        source_evidence=[
            {
                "field":
                    "affected_population",

                "source_quote": (
                    "affects 50 residents "
                    "after monsoon rain"
                ),
            },
            {
                "field":
                    "locality",

                "source_quote":
                    "Kanke",
            },
        ],
    )

    orchestrator = await AIOrchestrator.build(
        provider=MockProvider(
            structured_responses={
                ChallengeAnalysis:
                    payload,
            }
        )
    )

    result = await orchestrator.process_citizen_report(
        REPORT,
        context=CONTEXT,
    )

    assert (
        result.decision
        is ApprovalDecision.HUMAN_REVIEW
    )

    assert (
        result.problem_statement
        is None
    )

    assert any(
        "substantive civic-problem fact"
        in reason.casefold()
        for reason
        in result.decision_reasons
    )


@pytest.mark.asyncio
async def test_unknown_backend_context_is_not_sent_to_provider() -> None:
    def responder(
        prompt: str,
        response_model: type[ChallengeAnalysis],
    ) -> dict[str, object]:
        assert (
            "do-not-send-this-secret"
            not in prompt
        )

        assert (
            "internal_admin_note"
            not in prompt
        )

        return complete_analysis()

    orchestrator = await AIOrchestrator.build(
        provider=MockProvider(
            responder=responder
        )
    )

    context = {
        **CONTEXT,
        "internal_admin_note":
            "do-not-send-this-secret",
    }

    result = await orchestrator.process_citizen_report(
        REPORT,
        context=context,
    )

    assert (
        result.decision
        is ApprovalDecision.AUTO_APPROVED
    )


@pytest.mark.asyncio
async def test_uncertain_report_fails_closed_to_human_review() -> None:
    payload = complete_analysis(
        report_quality="uncertain",
        confidence="medium",
        is_actionable=False,
    )

    orchestrator = await AIOrchestrator.build(
        provider=MockProvider(
            structured_responses={
                ChallengeAnalysis:
                    payload,
            }
        )
    )

    result = await orchestrator.process_citizen_report(
        REPORT,
        context=CONTEXT,
    )

    assert (
        result.decision
        is ApprovalDecision.HUMAN_REVIEW
    )

    assert (
        result.problem_statement
        is None
    )

    assert any(
        "automatic decision"
        in reason.casefold()
        for reason
        in result.decision_reasons
    )


@pytest.mark.asyncio
async def test_incomplete_report_requests_citizen_revision() -> None:
    report = (
        "The drinking water has been dirty for several days "
        "and smells unusual."
    )

    payload = {
        "normalized_title":
            "The drinking water has been dirty",

        "summary":
            report,

        "category":
            "Water",

        "domain":
            "water",

        "report_quality":
            "incomplete",

        "is_sensible":
            True,

        "is_actionable":
            False,

        "missing_required_information": [
            "affected_water_source",
        ],

        "quality_reason": (
            "The problem is understandable, but the affected "
            "water source is not identified clearly enough."
        ),

        "confidence":
            "high",
    }

    orchestrator = await AIOrchestrator.build(
        provider=MockProvider(
            structured_responses={
                ChallengeAnalysis:
                    payload,
            }
        )
    )

    stages: list[AIStage] = []

    result = await orchestrator.process_citizen_report(
        report,
        context=LOCATION_CONTEXT,
        on_stage=stages.append,
    )

    assert (
        result.decision
        is ApprovalDecision.CITIZEN_REVISION_REQUIRED
    )

    assert (
        result.problem_statement
        is None
    )

    assert any(
        "affected water source"
        in reason.casefold()
        for reason
        in result.decision_reasons
    )

    assert (
        stages[-1]
        is AIStage.CITIZEN_REVISION_REQUIRED
    )


@pytest.mark.asyncio
async def test_high_confidence_gibberish_is_auto_rejected() -> None:
    report = (
        "asdf qwerty zxcv random meaningless words"
    )

    payload = {
        "normalized_title":
            "asdf qwerty zxcv",

        "summary":
            report,

        "report_quality":
            "gibberish",

        "is_sensible":
            False,

        "is_actionable":
            False,

        "confidence":
            "high",

        "quality_reason":
            "The submission contains no coherent civic problem.",

        "extracted_facts":
            {},

        "source_evidence":
            [],
    }

    orchestrator = await AIOrchestrator.build(
        provider=MockProvider(
            structured_responses={
                ChallengeAnalysis:
                    payload,
            }
        )
    )

    stages: list[AIStage] = []

    result = await orchestrator.process_citizen_report(
        report,
        context=LOCATION_CONTEXT,
        on_stage=stages.append,
    )

    assert (
        result.decision
        is ApprovalDecision.AUTO_REJECTED
    )

    assert (
        result.problem_statement
        is None
    )

    assert (
        stages[-1]
        is AIStage.REPORT_REJECTED
    )


@pytest.mark.asyncio
async def test_hindi_reports_produce_non_empty_embeddings() -> None:
    embedding = await EmbeddingService(
        HashingEmbeddingBackend()
    ).embed(
        "गाँव में पीने का पानी गंदा है"
    )

    assert any(
        value != 0
        for value
        in embedding.vector
    )

@pytest.mark.asyncio
async def test_human_finalizer_accepts_analysis_for_same_report() -> None:
    payload = complete_analysis(
        report_quality="uncertain",
        confidence="medium",
    )

    orchestrator = await AIOrchestrator.build(
        provider=MockProvider(
            structured_responses={
                ChallengeAnalysis: payload,
            }
        )
    )

    workflow = await orchestrator.process_citizen_report(
        REPORT,
        context=CONTEXT,
    )

    assert (
        workflow.decision
        is ApprovalDecision.HUMAN_REVIEW
    )

    statement = await orchestrator.finalize_human_approved_report(
        REPORT,
        workflow.analysis,
    )

    assert statement.citizen_report == REPORT


    @pytest.mark.asyncio
    async def test_human_finalizer_rejects_analysis_from_different_report() -> None:
        payload = complete_analysis(
            report_quality="uncertain",
            confidence="medium",
        )

        orchestrator = await AIOrchestrator.build(
            provider=MockProvider(
                structured_responses={
                    ChallengeAnalysis: payload,
                }
            )
        )

        workflow = await orchestrator.process_citizen_report(
            REPORT,
            context=CONTEXT,
        )

        different_report = (
            "A large pothole near the school gate "
            "is causing traffic problems."
        )

        with pytest.raises(
            ValueError,
            match="does not belong",
        ):
            await orchestrator.finalize_human_approved_report(
                different_report,
                workflow.analysis,
            )


    @pytest.mark.asyncio
    async def test_human_finalizer_rejects_unbound_analysis() -> None:
        payload = complete_analysis(
            report_quality="uncertain",
            confidence="medium",
        )

        orchestrator = await AIOrchestrator.build(
            provider=MockProvider(
                structured_responses={
                    ChallengeAnalysis: payload,
                }
            )
        )

        workflow = await orchestrator.process_citizen_report(
            REPORT,
            context=CONTEXT,
        )

        unbound_analysis = workflow.analysis.model_copy(
            update={
                "source_report_fingerprint": None,
            }
        )

        with pytest.raises(
            ValueError,
            match="does not belong",
        ):
            await orchestrator.finalize_human_approved_report(
                REPORT,
                unbound_analysis,
            )