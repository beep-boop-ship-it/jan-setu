from __future__ import annotations

import math
from dataclasses import dataclass, fields

from ai.embeddings.service import EmbeddingService
from ai.matching.explain import MatchExplainer
from ai.matching.features import ExtractedFeatures, MatchingFeatureExtractor
from ai.matching.retrieval import OrganizationRetriever
from ai.schemas.challenge import ConfidenceLevel
from ai.schemas.matching import (
    ChallengeRequirements,
    FeatureEvidence,
    MatchingResultSet,
    MatchResult,
    OrganizationCapabilityProfile,
)


@dataclass(frozen=True, slots=True)
class MatchingWeights:
    semantic: float = 0.22
    skills: float = 0.20
    domain: float = 0.15
    research: float = 0.13
    capacity: float = 0.08
    geography: float = 0.07
    experience: float = 0.10
    industry: float = 0.05

    def __post_init__(self) -> None:
        weights = self.as_dict()
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in weights.values()
        ):
            raise TypeError("Matching weights must be finite numbers")
        if any(not 0 <= value <= 1 for value in weights.values()):
            raise ValueError("Matching weights must be within 0..1")
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("Matching weights must sum to 1.0")

    def as_dict(self) -> dict[str, float]:
        return {item.name: float(getattr(self, item.name)) for item in fields(self)}


class WeightedMatchScorer:
    """Pure deterministic scorer; an LLM never controls the numeric result."""

    def __init__(self, weights: MatchingWeights | None = None) -> None:
        self.weights = weights or MatchingWeights()

    def score(
        self,
        challenge: ChallengeRequirements,
        organization: OrganizationCapabilityProfile,
        features: ExtractedFeatures,
    ) -> MatchResult:
        weights = self.weights.as_dict()
        expected = set(weights)
        actual = set(features.values)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"Feature set does not match scorer weights; missing={missing}, extra={extra}"
            )

        available_weight = sum(
            weights[name]
            for name, raw in features.values.items()
            if raw.available
        )
        if available_weight <= 0:
            raise ValueError("No applicable matching features were available")

        breakdown: list[FeatureEvidence] = []
        weighted_total = 0.0
        for name in weights:
            raw = features.values[name]
            effective_weight = weights[name] / available_weight if raw.available else 0.0
            contribution = raw.value * effective_weight
            weighted_total += contribution
            breakdown.append(
                FeatureEvidence(
                    name=name,
                    value=raw.value,
                    weight=effective_weight,
                    contribution=contribution,
                    positive_evidence=list(raw.positive_evidence),
                    gaps=list(raw.gaps),
                    available=raw.available,
                    data_present=raw.data_present,
                )
            )

        score = (
            0.0
            if features.hard_constraint_failures
            else max(0.0, min(1.0, weighted_total))
        )
        observed_weight = sum(
            weights[name]
            for name, raw in features.values.items()
            if raw.available and raw.data_present
        )
        coverage = observed_weight / available_weight
        confidence = (
            ConfidenceLevel.HIGH
            if coverage >= 0.9
            else ConfidenceLevel.MEDIUM
            if coverage >= 0.65
            else ConfidenceLevel.LOW
        )
        positives = [evidence for item in breakdown for evidence in item.positive_evidence]
        gaps = [gap for item in breakdown for gap in item.gaps]
        semantic = features.values["semantic"].value

        return MatchResult(
            challenge_id=challenge.id,
            organization_id=organization.id,
            organization_name=organization.name,
            final_score=score,
            semantic_score=semantic,
            feature_breakdown=breakdown,
            positive_evidence=positives,
            gaps=gaps,
            hard_constraint_failures=list(features.hard_constraint_failures),
            confidence=confidence,
        )


class MatchingService:
    """Coordinates retrieval, deterministic scoring, and safe explanations."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        retriever: OrganizationRetriever,
        feature_extractor: MatchingFeatureExtractor,
        scorer: WeightedMatchScorer,
        explainer: MatchExplainer,
        *,
        top_k: int = 10,
    ) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        self.embeddings = embeddings
        self.retriever = retriever
        self.feature_extractor = feature_extractor
        self.scorer = scorer
        self.explainer = explainer
        self.top_k = top_k

    async def match(
        self,
        challenge: ChallengeRequirements,
    ) -> MatchingResultSet:
        embedding = await self.embeddings.embed(challenge.embedding_text())
        candidates = await self.retriever.find_candidates(embedding, top_k=self.top_k)

        scored_matches: list[MatchResult] = []
        for candidate in candidates:
            if not candidate.organization.verified:
                continue
            features = self.feature_extractor.extract(
                challenge,
                candidate.organization,
                candidate.semantic_similarity,
            )
            scored_matches.append(
                self.scorer.score(challenge, candidate.organization, features)
            )

        scored_matches.sort(key=lambda item: (-item.final_score, item.organization_id))

        matches: list[MatchResult] = []
        for scored in scored_matches:
            matches.append(
                await self.explainer.explain(
                    scored
                )
            )

        return MatchingResultSet(
            challenge=challenge,
            matches=matches,
            candidates_considered=len(candidates),
        )
