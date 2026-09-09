from __future__ import annotations

from ai.dedup.classifier import HybridRelationshipClassifier
from ai.dedup.retrieval import ProblemRetriever
from ai.embeddings.service import EmbeddingService
from ai.schemas.dedup import (
    DecisionSource,
    DedupMatch,
    DedupResult,
    IncomingProblem,
    ProblemRelationship,
)


class DeduplicationService:
    """Retrieve similar reports and classify relationships without deleting reports."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        retriever: ProblemRetriever,
        classifier: HybridRelationshipClassifier,
        *,
        top_k: int = 10,
    ) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        self.embeddings = embeddings
        self.retriever = retriever
        self.classifier = classifier
        self.top_k = top_k

    async def find_duplicates(self, problem: IncomingProblem) -> DedupResult:
        embedding = await self.embeddings.embed(
            problem.embedding_text(),
            metadata={"problem_id": problem.id or ""},
        )
        candidates = await self.retriever.find_similar(
            embedding,
            top_k=self.top_k,
        )

        matches: list[DedupMatch] = []
        for candidate in candidates:
            signals, decision = await self.classifier.classify(
                problem,
                candidate.problem,
                candidate.vector_similarity,
            )
            matches.append(
                DedupMatch(
                    candidate=candidate.problem,
                    signals=signals,
                    classification=decision,
                )
            )

        priority = {
            ProblemRelationship.DUPLICATE: 0,
            ProblemRelationship.RELATED: 1,
            ProblemRelationship.INDEPENDENT: 2,
        }
        matches.sort(
            key=lambda item: (
                priority[item.classification.relationship],
                item.classification.requires_human_review,
                -item.classification.confidence,
                -item.signals.combined_signal,
                item.candidate.id or "",
            )
        )

        llm_used = any(
            item.classification.decision_source
            in {DecisionSource.LLM, DecisionSource.RULES_FALLBACK}
            for item in matches
        )
        needs_review = any(
            item.classification.requires_human_review for item in matches
        )

        return DedupResult(
            incoming_problem=problem,
            matches=matches,
            candidates_considered=len(candidates),
            llm_configured=self.classifier.provider is not None,
            llm_used=llm_used,
            needs_human_review=needs_review,
        )
