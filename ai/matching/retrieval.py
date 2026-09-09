from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ai.embeddings.models import Embedding
from ai.embeddings.service import cosine_similarity
from ai.schemas.matching import OrganizationCapabilityProfile, RetrievedOrganization


@runtime_checkable
class OrganizationRetriever(Protocol):
    async def find_candidates(
        self,
        embedding: Embedding,
        *,
        top_k: int,
    ) -> list[RetrievedOrganization]: ...


@dataclass(frozen=True, slots=True)
class IndexedOrganization:
    organization: OrganizationCapabilityProfile
    embedding: Embedding


class InMemoryOrganizationRetriever:
    def __init__(self, entries: list[IndexedOrganization] | None = None) -> None:
        self._entries = list(entries or [])
        self._validate_index()

    def _validate_index(self) -> None:
        ids = [entry.organization.id for entry in self._entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Organization IDs must be unique")
        if not self._entries:
            return
        reference = self._entries[0].embedding
        if any(entry.embedding.dimensions != reference.dimensions for entry in self._entries):
            raise ValueError("All organization embeddings must have equal dimensions")
        if any(entry.embedding.model != reference.model for entry in self._entries):
            raise ValueError("All organization embeddings must use the same model")

    def add(
        self,
        organization: OrganizationCapabilityProfile,
        embedding: Embedding,
    ) -> None:
        if any(entry.organization.id == organization.id for entry in self._entries):
            raise ValueError(f"Organization ID already indexed: {organization.id}")
        if self._entries:
            reference = self._entries[0].embedding
            if reference.dimensions != embedding.dimensions:
                raise ValueError("All organization embeddings must have equal dimensions")
            if reference.model != embedding.model:
                raise ValueError("All organization embeddings must use the same model")
        self._entries.append(IndexedOrganization(organization, embedding))

    async def find_candidates(
        self,
        embedding: Embedding,
        *,
        top_k: int,
    ) -> list[RetrievedOrganization]:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        if self._entries:
            reference = self._entries[0].embedding
            if embedding.dimensions != reference.dimensions:
                raise ValueError(
                    "Query embedding dimension does not match organization index"
                )
            if embedding.model != reference.model:
                raise ValueError(
                    "Query embedding model does not match organization index"
                )

        results = [
            RetrievedOrganization(
                organization=entry.organization,
                semantic_similarity=cosine_similarity(
                    embedding.vector,
                    entry.embedding.vector,
                ),
            )
            for entry in self._entries
            if entry.organization.verified
        ]
        return sorted(
            results,
            key=lambda item: (-item.semantic_similarity, item.organization.id),
        )[:top_k]
