import math
import socket

import pytest

from ai.embeddings.service import EmbeddingService, HashingEmbeddingBackend
from ai.matching.scorer import MatchingWeights
from ai.providers.base import ProviderResponseError
from ai.providers.mock import MockProvider
from ai.schemas.challenge import ChallengeAnalysis


def challenge_payload() -> dict:
    return {
        "normalized_title": "Seasonal drinking water contamination",
        "summary": "Drinking water becomes unsafe during monsoon.",
        "domain": "Water",
        "subdomain": "Water quality",
        "urgency": "high",
        "severity": "high",
        "required_capabilities": ["Water testing", "IoT sensing"],
        "skills": ["Water chemistry"],
        "unknown_fields": ["affected_population"],
        "fields_needing_confirmation": ["affected_population"],
        "confidence": "medium",
    }


@pytest.mark.asyncio
async def test_mock_provider_validates_without_network() -> None:
    provider = MockProvider(structured_responses={ChallengeAnalysis: challenge_payload()})
    result = await provider.generate_structured("ignored", ChallengeAnalysis)
    assert result.domain == "Water"
    assert provider.calls == [("structured", "ChallengeAnalysis")]


@pytest.mark.asyncio
async def test_mock_provider_never_touches_network(monkeypatch) -> None:
    def forbidden_socket(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    provider = MockProvider(structured_responses={ChallengeAnalysis: challenge_payload()})
    assert (await provider.generate_structured("offline", ChallengeAnalysis)).domain == "Water"


@pytest.mark.asyncio
async def test_mock_provider_rejects_malformed_response() -> None:
    provider = MockProvider(structured_responses={ChallengeAnalysis: {"summary": "missing title"}})
    with pytest.raises(ProviderResponseError):
        await provider.generate_structured("ignored", ChallengeAnalysis)


@pytest.mark.asyncio
async def test_hash_embeddings_are_deterministic_unit_vectors_and_batchable() -> None:
    service = EmbeddingService(HashingEmbeddingBackend())
    one = await service.embed("  Dirty WATER after rain! ")
    two, other = await service.embed_batch(["dirty water after rain", "school access"])
    assert one.vector == two.vector
    assert one.dimensions == 128
    assert len(other.vector) == 128
    assert math.isclose(sum(value * value for value in one.vector), 1.0)


def test_matching_weights_are_validated() -> None:
    with pytest.raises(ValueError):
        MatchingWeights(semantic=0.5)
