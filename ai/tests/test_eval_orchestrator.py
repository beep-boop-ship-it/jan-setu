import pytest

from ai.dedup.classifier import (
    HybridRelationshipClassifier,
)
from ai.embeddings.service import (
    EmbeddingService,
    HashingEmbeddingBackend,
)
from ai.eval.dedup_eval import (
    DedupEvaluator,
    LabelledProblemPair,
)
from ai.eval.matching_eval import (
    LabelledRanking,
    MatchingEvaluator,
)
from ai.orchestrator import (
    AIOrchestrator,
    AIStage,
)
from ai.providers.mock import MockProvider
from ai.schemas.challenge import ChallengeAnalysis
from ai.schemas.dedup import (
    IncomingProblem,
    ProblemRelationship,
)
from ai.schemas.matching import (
    ChallengeRequirements,
    MatchingResultSet,
    OrganizationCapabilityProfile,
)


@pytest.mark.asyncio
async def test_dedup_evaluator_calculates_metrics_from_labels() -> None:
    evaluator = DedupEvaluator(
        EmbeddingService(
            HashingEmbeddingBackend()
        ),
        HybridRelationshipClassifier(
            provider=None
        ),
    )

    duplicate_a = IncomingProblem(
        id="a",
        title="Broken road near market",
        description="Large pothole near the main market",
        locality="Ward 3",
        domain="road",
    )

    duplicate_b = IncomingProblem(
        id="b",
        title="Broken road near market",
        description="Large pothole near the main market",
        locality="Ward 3",
        domain="road",
    )

    independent_a = IncomingProblem(
        id="c",
        title="Contaminated drinking water",
        description="Drinking water smells bad",
        locality="Ward 5",
        domain="water",
    )

    independent_b = IncomingProblem(
        id="d",
        title="Streetlight not working",
        description="Streetlight is broken near the bus stop",
        locality="Ward 9",
        domain="electricity",
    )

    report = await evaluator.evaluate(
        [
            LabelledProblemPair(
                problem_a=duplicate_a,
                problem_b=duplicate_b,
                expected_relationship=(
                    ProblemRelationship.DUPLICATE
                ),
            ),
            LabelledProblemPair(
                problem_a=independent_a,
                problem_b=independent_b,
                expected_relationship=(
                    ProblemRelationship.INDEPENDENT
                ),
            ),
        ]
    )

    assert report.examples == 2

    assert (
        0 <= report.accuracy <= 1
    )

    assert (
        0 <= report.macro_f1 <= 1
    )

    assert (
        0 <= report.duplicate_precision <= 1
    )

    assert (
        0 <= report.duplicate_recall <= 1
    )

    assert (
        0 <= report.duplicate_f1 <= 1
    )

    assert sum(
        sum(
            predictions.values()
        )
        for predictions
        in report.confusion_matrix.values()
    ) == 2


def test_matching_evaluator_known_perfect_ranking() -> None:
    report = MatchingEvaluator().evaluate(
        [
            LabelledRanking(
                query_id="q1",
                ranked_organization_ids=[
                    "o1",
                    "o2",
                    "o3",
                ],
                relevance={
                    "o1": 3,
                    "o2": 2,
                    "o3": 1,
                },
            )
        ],
        k=3,
    )

    assert report.queries == 1
    assert report.k == 3

    assert (
        report.precision_at_k
        == pytest.approx(1.0)
    )

    assert (
        report.recall_at_k
        == pytest.approx(1.0)
    )

    assert (
        report.mrr
        == pytest.approx(1.0)
    )

    assert (
        report.ndcg_at_k
        == pytest.approx(1.0)
    )

    assert (
        report.pairwise_accuracy
        == pytest.approx(1.0)
    )

    assert (
        report.pairwise_pairs_evaluated
        == 3
    )

    assert not report.short_rankings


@pytest.mark.asyncio
async def test_orchestrator_wires_current_offline_services_and_progress() -> None:
    provider_analysis = {
        "normalized_title":
            "Water contamination",

        "summary":
            "Our water becomes unsafe after rain",

        "domain":
            "water",

        "required_capabilities": [
            "water testing",
        ],

        "skills": [
            "water testing",
        ],

        "unknown_fields": [
            "affected_population",
        ],

        "confidence":
            "medium",
    }

    provider = MockProvider(
        structured_responses={
            ChallengeAnalysis:
                provider_analysis,
        }
    )

    existing = IncomingProblem(
        id="p1",
        title="Unsafe water",
        description="Unsafe water after monsoon",
        locality="Ward 3",
        domain="water",
    )

    organization = OrganizationCapabilityProfile(
        id="o1",
        name="Water University",
        domains=[
            "water",
        ],
        skills=[
            "water testing",
        ],
        verified=True,
    )

    ai = await AIOrchestrator.build(
        provider=provider,
        problems=[
            existing,
        ],
        organizations=[
            organization,
        ],
    )

    stages: list[AIStage] = []

    result = await ai.analyze_challenge(
        "Our water becomes unsafe after rain",
        on_stage=stages.append,
    )

    assert (
        result.domain
        == "water"
    )

    assert stages == [
        AIStage.UNDERSTANDING_PROBLEM,
    ]

    duplicate_result = await ai.find_duplicates(
        IncomingProblem(
            title="Unsafe water",
            description="Unsafe water after monsoon",
            locality="Ward 3",
            domain="water",
        )
    )

    assert (
        duplicate_result.matches
    )

    matching_result = await ai.match_organizations(
        ChallengeRequirements(
            title="Unsafe water",
            summary="Unsafe water after monsoon",
            domain="water",
            skills=[
                "water testing",
            ],
            required_capabilities=[
                "water testing",
            ],
        )
    )

    assert isinstance(
        matching_result,
        MatchingResultSet,
    )

    assert (
        matching_result.candidates_considered
        == 1
    )

    assert (
        len(matching_result.matches)
        == 1
    )

    assert (
        matching_result.matches[0].organization_id
        == "o1"
    )
