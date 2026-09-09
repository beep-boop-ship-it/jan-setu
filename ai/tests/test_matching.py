import pytest

from ai.embeddings.models import EmbeddingConfig
from ai.embeddings.service import (
    EmbeddingService,
    HashingEmbeddingBackend,
)
from ai.matching.explain import MatchExplainer
from ai.matching.features import MatchingFeatureExtractor
from ai.matching.retrieval import InMemoryOrganizationRetriever
from ai.matching.scorer import (
    MatchingService,
    MatchingWeights,
    WeightedMatchScorer,
)
from ai.providers.mock import MockProvider
from ai.schemas.matching import (
    ChallengeRequirements,
    GeoPoint,
    OrganizationCapabilityProfile,
    PreviousProject,
)


def challenge() -> ChallengeRequirements:
    return ChallengeRequirements(
        id="c1",
        title="Seasonal drinking water contamination",
        summary="Unsafe water during monsoon",
        domain="water",
        skills=[
            "IoT sensing",
            "water chemistry",
        ],
        required_capabilities=[
            "water testing",
        ],
        location=GeoPoint(
            latitude=23.36,
            longitude=85.33,
        ),
    )


def organizations() -> list[OrganizationCapabilityProfile]:
    return [
        OrganizationCapabilityProfile(
            id="excellent",
            name="Water Innovation University",
            domains=[
                "water",
            ],
            skills=[
                "IoT sensing",
                "water chemistry",
                "water testing",
            ],
            research_areas=[
                "water testing",
            ],
            facilities=[
                "water lab",
            ],
            available_capacity=0.9,
            previous_projects=[
                PreviousProject(
                    title="Monsoon water sensors",
                    domains=[
                        "water",
                    ],
                    skills=[
                        "IoT sensing",
                    ],
                    verified=True,
                )
            ],
            location=GeoPoint(
                latitude=23.37,
                longitude=85.34,
            ),
            verified=True,
        ),
        OrganizationCapabilityProfile(
            id="partial",
            name="General Engineering College",
            domains=[
                "water",
            ],
            skills=[
                "IoT sensing",
            ],
            research_areas=[],
            facilities=[],
            available_capacity=0.3,
            location=GeoPoint(
                latitude=24.0,
                longitude=85.8,
            ),
            verified=True,
        ),
        OrganizationCapabilityProfile(
            id="mismatch",
            name="Arts Institute",
            domains=[
                "arts",
            ],
            skills=[
                "history",
            ],
            available_capacity=None,
            location=GeoPoint(
                latitude=28.6,
                longitude=77.2,
            ),
            verified=True,
        ),
    ]


async def matching_service(
    explainer: MatchExplainer | None = None,
) -> MatchingService:
    embeddings = EmbeddingService(
        HashingEmbeddingBackend(
            EmbeddingConfig(
                dimensions=256,
            )
        )
    )

    retriever = InMemoryOrganizationRetriever()

    for organization in organizations():
        vector = await embeddings.embed(
            organization.embedding_text()
        )

        retriever.add(
            organization,
            vector,
        )

    return MatchingService(
        embeddings,
        retriever,
        MatchingFeatureExtractor(),
        WeightedMatchScorer(),
        explainer or MatchExplainer(),
        top_k=10,
    )


@pytest.mark.asyncio
async def test_matching_is_deterministic_explainable_and_ranked() -> None:
    service = await matching_service()

    first = await service.match(
        challenge()
    )

    second = await service.match(
        challenge()
    )

    assert [
        item.organization_id
        for item in first.matches
    ] == [
        item.organization_id
        for item in second.matches
    ]

    assert [
        item.final_score
        for item in first.matches
    ] == [
        item.final_score
        for item in second.matches
    ]

    assert len(first.matches) == 3

    assert (
        first.matches[0].organization_id
        == "excellent"
    )

    assert (
        first.matches[0].final_score
        > first.matches[1].final_score
        > first.matches[2].final_score
    )

    assert {
        item.name
        for item
        in first.matches[0].feature_breakdown
    } == set(
        MatchingWeights().as_dict()
    )

    assert first.matches[0].explanation


@pytest.mark.asyncio
async def test_geography_and_capacity_are_handled_safely() -> None:
    service = await matching_service()

    result = await service.match(
        challenge()
    )

    excellent = next(
        item
        for item in result.matches
        if item.organization_id == "excellent"
    )

    features = {
        item.name: item
        for item in excellent.feature_breakdown
    }

    assert (
        features["geography"].value
        > 0.9
    )

    assert (
        features["capacity"].value
        == pytest.approx(0.9)
    )

    mismatch = next(
        item
        for item in result.matches
        if item.organization_id == "mismatch"
    )

    mismatch_features = {
        item.name: item
        for item in mismatch.feature_breakdown
    }

    assert (
        "capacity"
        in mismatch_features
    )


@pytest.mark.asyncio
async def test_unverified_organizations_are_excluded() -> None:
    embeddings = EmbeddingService(
        HashingEmbeddingBackend(
            EmbeddingConfig(
                dimensions=256,
            )
        )
    )

    retriever = InMemoryOrganizationRetriever()

    unverified = OrganizationCapabilityProfile(
        id="unverified",
        name="Unverified Water Lab",
        domains=[
            "water",
        ],
        skills=[
            "water testing",
        ],
        verified=False,
    )

    vector = await embeddings.embed(
        unverified.embedding_text()
    )

    retriever.add(
        unverified,
        vector,
    )

    service = MatchingService(
        embeddings,
        retriever,
        MatchingFeatureExtractor(),
        WeightedMatchScorer(),
        MatchExplainer(),
        top_k=10,
    )

    result = await service.match(
        challenge()
    )

    assert all(
        item.organization_id != "unverified"
        for item in result.matches
    )


@pytest.mark.asyncio
async def test_hard_constraints_zero_score() -> None:
    constrained = challenge().model_copy(
        update={
            "required_facilities": [
                "Biosafety level 3 lab",
            ]
        }
    )

    service = await matching_service()

    result = await service.match(
        constrained
    )

    assert result.matches

    assert all(
        item.final_score == 0
        for item in result.matches
    )

    assert all(
        item.hard_constraint_failures
        for item in result.matches
    )


@pytest.mark.asyncio
async def test_llm_explanation_cannot_change_score() -> None:
    provider = MockProvider(
        text_responses=[
            (
                "Polished explanation claiming 100%, "
                "but it is only text."
            )
        ]
        * 3
    )

    service = await matching_service(
        MatchExplainer(
            provider
        )
    )

    plain = await service.match(
        challenge(),
        polish_explanations=False,
    )

    polished = await service.match(
        challenge(),
        polish_explanations=True,
    )

    assert [
        item.final_score
        for item in plain.matches
    ] == [
        item.final_score
        for item in polished.matches
    ]


def test_scoring_configuration_validation() -> None:
    with pytest.raises(ValueError):
        MatchingWeights(
            semantic=-0.1,
            skills=0.3,
        )
