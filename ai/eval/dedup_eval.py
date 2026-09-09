from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ai.dedup.classifier import HybridRelationshipClassifier
from ai.embeddings.service import EmbeddingService, cosine_similarity
from ai.schemas.dedup import IncomingProblem, ProblemRelationship


class LabelledProblemPair(BaseModel):
    """One human-labelled pair used to evaluate deduplication."""

    model_config = ConfigDict(extra="forbid")

    problem_a: IncomingProblem
    problem_b: IncomingProblem
    expected_relationship: ProblemRelationship


class ClassMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    support: int = Field(ge=0)


class DedupEvaluationReport(BaseModel):
    """Reproducible metrics for semantic deduplication and auto-cluster safety."""

    model_config = ConfigDict(extra="forbid")

    examples: int = Field(gt=0)
    embedding_model: str = Field(min_length=1, max_length=200)
    classifier_mode: str = Field(min_length=1, max_length=200)
    thresholds: dict[str, float] = Field(default_factory=dict)

    accuracy: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    per_class: dict[str, ClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]

    # Backward-compatible duplicate-class metrics.
    duplicate_precision: float = Field(ge=0, le=1)
    duplicate_recall: float = Field(ge=0, le=1)
    duplicate_f1: float = Field(ge=0, le=1)

    # Safety metrics for automatic clustering. These count only decisions
    # explicitly marked auto_cluster_eligible by the hardened dedup schema.
    auto_cluster_precision: float = Field(ge=0, le=1)
    auto_cluster_recall: float = Field(ge=0, le=1)
    auto_cluster_false_positives: list[int] = Field(default_factory=list)

    false_positives: list[int] = Field(default_factory=list)
    false_negatives: list[int] = Field(default_factory=list)
    human_review_cases: list[int] = Field(default_factory=list)
    borderline_cases: list[int] = Field(default_factory=list)


class DedupEvaluator:
    def __init__(
        self,
        embeddings: EmbeddingService,
        classifier: HybridRelationshipClassifier,
    ) -> None:
        self.embeddings = embeddings
        self.classifier = classifier

    async def evaluate(
        self,
        examples: list[LabelledProblemPair],
        *,
        borderline_margin: float = 0.05,
    ) -> DedupEvaluationReport:
        if not examples:
            raise ValueError(
                "Labelled examples are required; metrics are never fabricated"
            )
        if (
            isinstance(borderline_margin, bool)
            or not isinstance(borderline_margin, (int, float))
            or not math.isfinite(float(borderline_margin))
        ):
            raise TypeError("borderline_margin must be a finite number")
        if not 0 <= float(borderline_margin) <= 1:
            raise ValueError("borderline_margin must be within 0..1")

        relations = list(ProblemRelationship)
        relation_keys = [relationship.value for relationship in relations]
        confusion = {
            expected: {predicted: 0 for predicted in relation_keys}
            for expected in relation_keys
        }

        correct = 0
        duplicate_tp = duplicate_fp = duplicate_fn = 0
        auto_tp = auto_fp = 0

        false_positives: list[int] = []
        false_negatives: list[int] = []
        auto_cluster_false_positives: list[int] = []
        human_review_cases: list[int] = []
        borderline_cases: list[int] = []

        related_threshold = self.classifier.thresholds.related_similarity
        duplicate_threshold = self.classifier.thresholds.duplicate_similarity
        margin = float(borderline_margin)

        for index, example in enumerate(examples):
            left, right = await self.embeddings.embed_batch(
                [
                    example.problem_a.embedding_text(),
                    example.problem_b.embedding_text(),
                ]
            )
            similarity = cosine_similarity(left.vector, right.vector)
            signals, decision = await self.classifier.classify(
                example.problem_a,
                example.problem_b,
                similarity,
            )

            expected = example.expected_relationship
            predicted = decision.relationship
            confusion[expected.value][predicted.value] += 1

            if predicted is expected:
                correct += 1

            predicted_duplicate = predicted is ProblemRelationship.DUPLICATE
            actual_duplicate = expected is ProblemRelationship.DUPLICATE

            if predicted_duplicate and actual_duplicate:
                duplicate_tp += 1
            elif predicted_duplicate:
                duplicate_fp += 1
                false_positives.append(index)
            elif actual_duplicate:
                duplicate_fn += 1
                false_negatives.append(index)

            if decision.auto_cluster_eligible:
                if actual_duplicate:
                    auto_tp += 1
                else:
                    auto_fp += 1
                    auto_cluster_false_positives.append(index)

            if decision.requires_human_review:
                human_review_cases.append(index)

            near_related_boundary = (
                abs(signals.combined_signal - related_threshold) <= margin
            )
            near_duplicate_boundary = (
                abs(signals.combined_signal - duplicate_threshold) <= margin
            )
            if (
                decision.requires_human_review
                or near_related_boundary
                or near_duplicate_boundary
            ):
                borderline_cases.append(index)

        per_class: dict[str, ClassMetrics] = {}
        for relationship in relations:
            key = relationship.value
            tp = confusion[key][key]
            fp = sum(
                confusion[expected][key]
                for expected in relation_keys
                if expected != key
            )
            fn = sum(
                confusion[key][predicted]
                for predicted in relation_keys
                if predicted != key
            )
            support = sum(confusion[key].values())

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            per_class[key] = ClassMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            )

        duplicate_precision = (
            duplicate_tp / (duplicate_tp + duplicate_fp)
            if duplicate_tp + duplicate_fp
            else 0.0
        )
        duplicate_recall = (
            duplicate_tp / (duplicate_tp + duplicate_fn)
            if duplicate_tp + duplicate_fn
            else 0.0
        )
        duplicate_f1 = (
            2
            * duplicate_precision
            * duplicate_recall
            / (duplicate_precision + duplicate_recall)
            if duplicate_precision + duplicate_recall
            else 0.0
        )

        actual_duplicates = sum(
            example.expected_relationship is ProblemRelationship.DUPLICATE
            for example in examples
        )
        auto_cluster_precision = (
            auto_tp / (auto_tp + auto_fp)
            if auto_tp + auto_fp
            else 0.0
        )
        auto_cluster_recall = (
            auto_tp / actual_duplicates if actual_duplicates else 0.0
        )

        provider = self.classifier.provider
        classifier_mode = (
            "rules_only"
            if provider is None
            else f"hybrid:{provider.name}"
        )
        embedding_model = getattr(
            self.embeddings,
            "model",
            getattr(self.embeddings.backend, "model", "unknown"),
        )

        return DedupEvaluationReport(
            examples=len(examples),
            embedding_model=str(embedding_model),
            classifier_mode=classifier_mode,
            thresholds=self.classifier.thresholds.as_dict(),
            accuracy=correct / len(examples),
            macro_f1=(
                sum(
                    metric.f1
                    for metric in per_class.values()
                    if metric.support > 0
                )
                / sum(
                    1
                    for metric in per_class.values()
                    if metric.support > 0
                )
            ),
            per_class=per_class,
            confusion_matrix=confusion,
            duplicate_precision=duplicate_precision,
            duplicate_recall=duplicate_recall,
            duplicate_f1=duplicate_f1,
            auto_cluster_precision=auto_cluster_precision,
            auto_cluster_recall=auto_cluster_recall,
            auto_cluster_false_positives=auto_cluster_false_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            human_review_cases=human_review_cases,
            borderline_cases=borderline_cases,
        )

    async def evaluate_thresholds(
        self,
        examples: list[LabelledProblemPair],
        threshold_pairs: list[tuple[float, float]],
        *,
        borderline_margin: float = 0.05,
    ) -> dict[str, DedupEvaluationReport]:
        """
        Evaluate (duplicate, related) thresholds deterministically.

        The provider is deliberately disabled during threshold sweeps so the
        result measures rule thresholds rather than LLM variability or API state.
        """
        if not threshold_pairs:
            raise ValueError("At least one threshold pair is required")

        reports: dict[str, DedupEvaluationReport] = {}

        for duplicate, related in threshold_pairs:
            thresholds = replace(
                self.classifier.thresholds,
                duplicate_similarity=duplicate,
                related_similarity=related,
            )
            evaluator = DedupEvaluator(
                self.embeddings,
                HybridRelationshipClassifier(
                    thresholds=thresholds,
                    provider=None,
                ),
            )
            key = f"duplicate={duplicate:.3f},related={related:.3f}"
            reports[key] = await evaluator.evaluate(
                examples,
                borderline_margin=borderline_margin,
            )

        return reports


def _load_dataset(path: Path) -> list[LabelledProblemPair]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read labelled dataset: {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError("Dedup dataset JSON root must be an array")

    examples = [LabelledProblemPair.model_validate(item) for item in raw]
    if not examples:
        raise ValueError("Dedup dataset must contain at least one labelled pair")

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate JanSetu deduplication on labelled JSON pairs. "
            "The default hashing embedding is an offline baseline, not a "
            "production semantic-embedding benchmark."
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--duplicate-threshold", type=float)
    parser.add_argument("--related-threshold", type=float)
    parser.add_argument("--borderline-margin", type=float, default=0.05)
    args = parser.parse_args()

    from ai.embeddings.service import HashingEmbeddingBackend

    examples = _load_dataset(args.dataset)
    classifier = HybridRelationshipClassifier(provider=None)

    if (
        args.duplicate_threshold is not None
        or args.related_threshold is not None
    ):
        thresholds = replace(
            classifier.thresholds,
            duplicate_similarity=(
                classifier.thresholds.duplicate_similarity
                if args.duplicate_threshold is None
                else args.duplicate_threshold
            ),
            related_similarity=(
                classifier.thresholds.related_similarity
                if args.related_threshold is None
                else args.related_threshold
            ),
        )
        classifier = HybridRelationshipClassifier(
            thresholds=thresholds,
            provider=None,
        )

    evaluator = DedupEvaluator(
        EmbeddingService(HashingEmbeddingBackend()),
        classifier,
    )
    report = asyncio.run(
        evaluator.evaluate(
            examples,
            borderline_margin=args.borderline_margin,
        )
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
