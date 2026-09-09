from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class LabelledRanking(BaseModel):
    """Human relevance judgements for one organization-ranking query."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1, max_length=128)
    ranked_organization_ids: list[str] = Field(
        default_factory=list,
        max_length=1_000,
    )
    relevance: dict[str, float] = Field(
        min_length=1,
        max_length=1_000,
        description=(
            "Organization ID to finite non-negative relevance grade. "
            "At least one organization must have positive relevance."
        ),
    )

    @field_validator("query_id", mode="after")
    @classmethod
    def clean_query_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query_id must not be empty")
        return cleaned

    @field_validator("ranked_organization_ids", mode="after")
    @classmethod
    def validate_ranked_ids(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        for value in values:
            if not isinstance(value, str):
                raise TypeError("ranked organization IDs must be strings")

            item = value.strip()
            if not item:
                raise ValueError("ranked organization IDs must not be empty")
            if len(item) > 128:
                raise ValueError(
                    "ranked organization IDs must not exceed 128 characters"
                )
            if item in seen:
                raise ValueError(
                    f"Duplicate ranked organization ID: {item!r}"
                )

            seen.add(item)
            cleaned.append(item)

        return cleaned

    @field_validator("relevance", mode="before")
    @classmethod
    def validate_relevance(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise TypeError("relevance must be an object mapping organization IDs to grades")
        cleaned: dict[str, float] = {}

        for raw_id, raw_grade in value.items():
            if not isinstance(raw_id, str):
                raise TypeError("relevance organization IDs must be strings")

            organization_id = raw_id.strip()
            if not organization_id:
                raise ValueError("relevance organization IDs must not be empty")
            if len(organization_id) > 128:
                raise ValueError(
                    "relevance organization IDs must not exceed 128 characters"
                )

            if isinstance(raw_grade, bool) or not isinstance(
                raw_grade,
                (int, float),
            ):
                raise TypeError("relevance grades must be numeric")

            grade = float(raw_grade)
            if not math.isfinite(grade):
                raise ValueError("relevance grades must be finite")
            if grade < 0:
                raise ValueError("relevance grades must be non-negative")
            if grade > 100:
                raise ValueError("relevance grades must not exceed 100")

            if organization_id in cleaned:
                raise ValueError(
                    f"Duplicate relevance organization ID after trimming: {organization_id!r}"
                )

            cleaned[organization_id] = grade

        return cleaned

    @model_validator(mode="after")
    def require_positive_relevance(self) -> "LabelledRanking":
        if not any(grade > 0 for grade in self.relevance.values()):
            raise ValueError(
                "Each evaluation query must contain at least one relevant organization"
            )
        return self


class MatchingEvaluationReport(BaseModel):
    """Ranking metrics for JanSetu organization matching only."""

    model_config = ConfigDict(extra="forbid")

    evaluation_scope: str = "organization_ranking"
    queries: int = Field(gt=0)
    k: int = Field(gt=0)
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)
    pairwise_accuracy: float = Field(ge=0, le=1)
    pairwise_pairs_evaluated: int = Field(ge=0)
    short_rankings: list[str] = Field(default_factory=list)


class MatchingEvaluator:
    def evaluate(
        self,
        examples: list[LabelledRanking],
        *,
        k: int = 5,
    ) -> MatchingEvaluationReport:
        if not examples:
            raise ValueError(
                "Labelled rankings are required; metrics are never fabricated"
            )
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 1:
            raise ValueError("k must be positive")

        query_ids = [example.query_id for example in examples]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be unique across the dataset")

        precisions: list[float] = []
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []
        short_rankings: list[str] = []
        pair_correct = 0
        pair_total = 0

        for example in examples:
            relevant = {
                organization_id
                for organization_id, grade in example.relevance.items()
                if grade > 0
            }
            top = example.ranked_organization_ids[:k]

            if len(top) < k:
                short_rankings.append(example.query_id)

            hits = sum(item in relevant for item in top)

            # Deliberately divide by k rather than len(top): failing to return
            # enough recommendations counts against a top-k ranking system.
            precisions.append(hits / k)
            recalls.append(hits / len(relevant))

            first_relevant_rank = next(
                (
                    index + 1
                    for index, item in enumerate(
                        example.ranked_organization_ids
                    )
                    if item in relevant
                ),
                None,
            )
            reciprocal_ranks.append(
                1 / first_relevant_rank
                if first_relevant_rank is not None
                else 0.0
            )

            dcg = sum(
                (2.0 ** example.relevance.get(item, 0.0) - 1.0)
                / math.log2(index + 2)
                for index, item in enumerate(top)
            )
            ideal_grades = sorted(
                example.relevance.values(),
                reverse=True,
            )[:k]
            idcg = sum(
                (2.0 ** grade - 1.0) / math.log2(index + 2)
                for index, grade in enumerate(ideal_grades)
            )
            ndcgs.append(dcg / idcg if idcg else 0.0)

            ranking_position = {
                organization_id: index
                for index, organization_id in enumerate(
                    example.ranked_organization_ids
                )
            }
            judged_ids = list(example.relevance)

            for index, left in enumerate(judged_ids):
                for right in judged_ids[index + 1 :]:
                    left_grade = example.relevance[left]
                    right_grade = example.relevance[right]

                    if left_grade == right_grade:
                        continue

                    left_position = ranking_position.get(left)
                    right_position = ranking_position.get(right)

                    # If neither judged item was returned, the system expressed
                    # no ordering between them, so this pair is not scoreable.
                    if left_position is None and right_position is None:
                        continue

                    pair_total += 1
                    expected_left_before_right = left_grade > right_grade

                    if left_position is None:
                        observed_left_before_right = False
                    elif right_position is None:
                        observed_left_before_right = True
                    else:
                        observed_left_before_right = (
                            left_position < right_position
                        )

                    if (
                        expected_left_before_right
                        == observed_left_before_right
                    ):
                        pair_correct += 1

        count = len(examples)

        return MatchingEvaluationReport(
            queries=count,
            k=k,
            precision_at_k=sum(precisions) / count,
            recall_at_k=sum(recalls) / count,
            mrr=sum(reciprocal_ranks) / count,
            ndcg_at_k=sum(ndcgs) / count,
            pairwise_accuracy=(
                pair_correct / pair_total if pair_total else 0.0
            ),
            pairwise_pairs_evaluated=pair_total,
            short_rankings=short_rankings,
        )


def _load_dataset(path: Path) -> list[LabelledRanking]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read labelled dataset: {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError("Matching dataset JSON root must be an array")

    examples = [LabelledRanking.model_validate(item) for item in raw]
    if not examples:
        raise ValueError(
            "Matching dataset must contain at least one labelled ranking"
        )

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate JanSetu organization-ranking outputs from labelled JSON. "
            "These metrics do not represent student/researcher/mentor matching."
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    examples = _load_dataset(args.dataset)
    report = MatchingEvaluator().evaluate(examples, k=args.k)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
