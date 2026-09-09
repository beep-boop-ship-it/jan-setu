from datetime import datetime, timedelta, timezone

import pytest

from ai.challenge.analyzer import ChallengeAnalyzer
from ai.dedup.classifier import (
    DedupThresholds,
    HybridRelationshipClassifier,
)
from ai.dedup.retrieval import InMemoryProblemRetriever
from ai.dedup.service import DeduplicationService
from ai.embeddings.models import EmbeddingConfig
from ai.embeddings.service import (
    EmbeddingService,
    HashingEmbeddingBackend,
)
from ai.providers.base import ProviderResponseError
from ai.providers.mock import MockProvider
from ai.schemas.challenge import ChallengeAnalysis
from ai.schemas.dedup import (
    IncomingProblem,
    ProblemRelationship,
)


def analysis_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "normalized_title": (
            "Seasonal drinking water contamination"
        ),
        "summary": (
            "Village drinking water becomes unsafe "
            "during monsoon."
        ),
        "domain": "Water",
        "subdomain": "Water quality",
        "urgency": "high",
        "severity": "high",
        "affected_population": None,
        "required_capabilities": [
            "Water testing",
            "IoT sensing",
        ],
        "skills": [
            "Water chemistry",
            "Civil engineering",
        ],
        "unknown_fields": [
            "affected_population",
        ],
        "fields_needing_confirmation": [
            "affected_population",
        ],
        "confidence": "medium",
    }

    payload.update(updates)
    return payload


@pytest.mark.asyncio
async def test_challenge_analyzer_returns_valid_grounded_analysis() -> None:
    provider = MockProvider(
        structured_responses={
            ChallengeAnalysis: analysis_payload(),
        }
    )

    result = await ChallengeAnalyzer(
        provider
    ).analyze(
        "Water becomes unsafe every monsoon"
    )

    assert (
        result.review_status
        == "needs_review"
    )

    assert result.affected_population is None

    assert (
        "affected_population"
        in result.unknown_fields
    )

    # Free-form model inferences are intentionally discarded.
    assert result.inferences == []


@pytest.mark.asyncio
async def test_challenge_analyzer_removes_unsupported_population_guess() -> None:
    provider = MockProvider(
        structured_responses={
            ChallengeAnalysis: analysis_payload(
                affected_population=438
            ),
        }
    )

    result = await ChallengeAnalyzer(
        provider
    ).analyze(
        "Water becomes unsafe every monsoon"
    )

    assert result.affected_population is None

    assert (
        "affected_population"
        in result.fields_needing_confirmation
    )

    assert any(
        "affected population"
        in issue.casefold()
        for issue in result.grounding_issues
    )

    # Unsupported factual guesses must not survive as explanatory
    # inference objects.
    assert result.inferences == []


@pytest.mark.asyncio
async def test_challenge_analyzer_surfaces_malformed_provider_response() -> None:
    provider = MockProvider(
        structured_responses={
            ChallengeAnalysis: {
                "summary": "bad",
            }
        }
    )

    with pytest.raises(
        ProviderResponseError
    ):
        await ChallengeAnalyzer(
            provider
        ).analyze(
            "A sufficiently clear citizen report"
        )


async def build_dedup_service(
    existing: list[IncomingProblem],
) -> DeduplicationService:
    embeddings = EmbeddingService(
        HashingEmbeddingBackend(
            EmbeddingConfig(
                dimensions=256,
            )
        )
    )

    retriever = InMemoryProblemRetriever()

    for item in existing:
        vector = await embeddings.embed(
            item.embedding_text(),
            metadata={
                "problem_id": item.id or "",
            },
        )

        retriever.add(
            item,
            vector,
        )

    thresholds = DedupThresholds(
        duplicate_similarity=0.78,
        related_similarity=0.30,
    )

    return DeduplicationService(
        embeddings,
        retriever,
        HybridRelationshipClassifier(
            thresholds
        ),
        top_k=10,
    )


@pytest.mark.asyncio
async def test_dedup_operates_without_llm_for_related_and_unrelated_reports() -> None:
    water = IncomingProblem(
        id="water-1",
        title="Seasonal water contamination",
        description=(
            "Dirty drinking water in Village X "
            "during monsoon"
        ),
        domain="water",
        tags=[
            "water quality",
            "monsoon",
        ],
        locality="Village X",
    )

    school = IncomingProblem(
        id="school-1",
        title="School roof repair",
        description=(
            "The primary school roof leaks "
            "and needs repair"
        ),
        domain="education",
        tags=[
            "school",
            "infrastructure",
        ],
        locality="Village Y",
    )

    service = await build_dedup_service(
        [
            water,
            school,
        ]
    )

    incoming = IncomingProblem(
        title="Water becomes dirty after rain",
        description=(
            "Seasonal drinking water contamination "
            "during monsoon"
        ),
        domain="water",
        tags=[
            "water quality",
            "monsoon",
        ],
        locality="Village X",
    )

    result = await service.find_duplicates(
        incoming
    )

    by_id = {
        item.candidate.id:
            item.classification.relationship
        for item in result.matches
    }

    assert by_id["water-1"] in {
        ProblemRelationship.DUPLICATE,
        ProblemRelationship.RELATED,
    }

    assert (
        by_id["school-1"]
        is ProblemRelationship.INDEPENDENT
    )

    assert result.llm_available is False


@pytest.mark.asyncio
async def test_same_incident_with_matching_locality_can_be_duplicate() -> None:
    reported_at = datetime(
        2026,
        8,
        1,
        tzinfo=timezone.utc,
    )

    existing = IncomingProblem(
        id="1",
        title="Water smells after rain",
        description="Water smells after rain",
        domain="water",
        locality="Ward 4",
        reported_at=reported_at,
    )

    service = await build_dedup_service(
        [existing]
    )

    incoming = IncomingProblem(
        title="Water smells after rain",
        description="Water smells after rain",
        domain="water",
        locality="Ward 4",
        reported_at=(
            reported_at
            + timedelta(days=2)
        ),
    )

    result = await service.find_duplicates(
        incoming
    )

    assert result.matches

    assert (
        result.matches[0]
        .classification
        .relationship
        is ProblemRelationship.DUPLICATE
    )


@pytest.mark.asyncio
async def test_same_wording_far_apart_in_time_is_not_auto_duplicate() -> None:
    first_date = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    existing = IncomingProblem(
        id="old-water-incident",
        title="Water smells after rain",
        description="Water smells after rain",
        domain="water",
        locality="Ward 4",
        reported_at=first_date,
    )

    service = await build_dedup_service(
        [existing]
    )

    recurring = IncomingProblem(
        title="Water smells after rain",
        description="Water smells after rain",
        domain="water",
        locality="Ward 4",
        reported_at=(
            first_date
            + timedelta(days=90)
        ),
    )

    result = await service.find_duplicates(
        recurring
    )

    assert result.matches

    classification = (
        result.matches[0].classification
    )

    assert (
        classification.relationship
        is not ProblemRelationship.DUPLICATE
    )

    assert (
        classification.relationship
        is ProblemRelationship.RELATED
    )


@pytest.mark.asyncio
async def test_empty_dedup_index_is_safe() -> None:
    empty = await build_dedup_service(
        []
    )

    result = await empty.find_duplicates(
        IncomingProblem(
            title="New civic report",
            description="A complete civic problem report",
            domain="water",
            locality="Ward 1",
        )
    )

    assert result.matches == []


def test_threshold_boundaries_are_centralized() -> None:
    classifier = HybridRelationshipClassifier(
        DedupThresholds(
            duplicate_similarity=0.8,
            related_similarity=0.5,
        )
    )

    a = IncomingProblem(
        title="A water report",
        description="water",
        domain="water",
        locality="x",
    )

    b = IncomingProblem(
        title="A water report",
        description="different",
        domain="water",
        locality="x",
    )

    signals = classifier.signals(
        a,
        b,
        0.67,
    )

    assert (
        signals.combined_signal
        == pytest.approx(0.8)
    )
