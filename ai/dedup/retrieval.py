from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ai.embeddings.models import Embedding
from ai.embeddings.service import cosine_similarity
from ai.schemas.dedup import IncomingProblem, RetrievedProblem


@runtime_checkable
class ProblemRetriever(Protocol):
    async def find_similar(
        self,
        embedding: Embedding,
        *,
        top_k: int,
    ) -> list[RetrievedProblem]: ...


@dataclass(frozen=True, slots=True)
class IndexedProblem:
    problem: IncomingProblem
    embedding: Embedding


class InMemoryProblemRetriever:
    """Replaceable demo adapter; production can implement this port with pgvector."""

    def __init__(self, entries: list[IndexedProblem] | None = None) -> None:
        self._entries: list[IndexedProblem] = []
        for entry in entries or []:
            self.add(entry.problem, entry.embedding)

    def add(self, problem: IncomingProblem, embedding: Embedding) -> None:
        if not problem.id:
            raise ValueError("Indexed problems must have a non-empty ID")

        if any(entry.problem.id == problem.id for entry in self._entries):
            raise ValueError(f"Problem ID {problem.id!r} is already indexed")

        if self._entries:
            reference = self._entries[0].embedding
            if reference.dimensions != embedding.dimensions:
                raise ValueError("All indexed problem embeddings must have equal dimensions")
            if reference.model != embedding.model:
                raise ValueError("All indexed problem embeddings must use the same model")

        self._entries.append(IndexedProblem(problem, embedding))

    async def find_similar(
        self,
        embedding: Embedding,
        *,
        top_k: int,
    ) -> list[RetrievedProblem]:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        if not self._entries:
            return []

        reference = self._entries[0].embedding
        if embedding.dimensions != reference.dimensions:
            raise ValueError(
                "Query embedding dimensions do not match the problem index"
            )
        if embedding.model != reference.model:
            raise ValueError(
                "Query embedding model does not match the problem index"
            )

        query_problem_id = str(embedding.metadata.get("problem_id", "")).strip()

        candidates = [
            RetrievedProblem(
                problem=entry.problem,
                vector_similarity=cosine_similarity(
                    embedding.vector,
                    entry.embedding.vector,
                ),
            )
            for entry in self._entries
            if entry.problem.id != query_problem_id
        ]

        return sorted(
            candidates,
            key=lambda item: (
                -item.vector_similarity,
                item.problem.id or "",
                item.problem.title.casefold(),
            ),
        )[:top_k]
