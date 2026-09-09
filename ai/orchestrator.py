from __future__ import annotations

import inspect
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from ai.challenge.analyzer import (
    ChallengeAnalyzer,
    report_fingerprint,
)
from ai.image_evidence.analyzer import (
    ImageEvidenceAnalyzer,
    ImageEvidenceProvider,
)
from ai.dedup.classifier import (
    DedupThresholds,
    HybridRelationshipClassifier,
)
from ai.dedup.retrieval import InMemoryProblemRetriever
from ai.dedup.service import DeduplicationService
from ai.embeddings.models import EmbeddingConfig
from ai.embeddings.service import (
    EmbeddingBackend,
    EmbeddingService,
    HashingEmbeddingBackend,
)
from ai.matching.explain import MatchExplainer
from ai.matching.features import MatchingFeatureExtractor
from ai.matching.participants import ParticipantMatchingService
from ai.matching.retrieval import InMemoryOrganizationRetriever
from ai.matching.scorer import (
    MatchingService,
    MatchingWeights,
    WeightedMatchScorer,
)
from ai.providers.base import (
    AIProvider,
    FallbackProvider,
    ProviderConfig,
)
from ai.providers.mock import MockProvider
from ai.schemas.challenge import (
    ChallengeAnalysis,
    ConfidenceLevel,
    ReportQuality,
    ReviewStatus,
)
from ai.schemas.image_evidence import (
    ImageEvidenceAnalysis,
    ImageEvidenceStatus,
)
from ai.schemas.dedup import DedupResult, IncomingProblem
from ai.schemas.matching import (
    ChallengeRequirements,
    MatchingResultSet,
    OrganizationCapabilityProfile,
)
from ai.schemas.participants import (
    ParticipantProfile,
    ParticipantRoutingResult,
)
from ai.schemas.workflow import (
    ApprovalDecision,
    ProblemStatement,
    ReportWorkflowResult,
)


logger = logging.getLogger(__name__)

_REQUIRED_CITIZEN_LOCATION_FIELDS = (
    "locality",
    "district",
    "pin_code",
)

_PIN_CODE_PATTERN = re.compile(r"^\d{6}$")

_MAX_DECISION_REASONS = 20


class AIProcessingError(RuntimeError):
    """Raised when AI processing cannot complete safely."""


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    """Deterministic checks required before automatic approval."""

    min_report_characters: int = 20
    require_location: bool = True
    require_source_evidence: bool = True
    require_core_evidence: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_report_characters, bool)
            or not isinstance(self.min_report_characters, int)
        ):
            raise TypeError(
                "min_report_characters must be an integer"
            )

        if self.min_report_characters < 3:
            raise ValueError(
                "min_report_characters must be at least 3"
            )

        for name in (
            "require_location",
            "require_source_evidence",
            "require_core_evidence",
        ):
            if not isinstance(
                getattr(self, name),
                bool,
            ):
                raise TypeError(
                    f"{name} must be a boolean"
                )


@dataclass(frozen=True, slots=True)
class AIConfig:
    provider: ProviderConfig = field(
        default_factory=ProviderConfig.from_env
    )
    embedding: EmbeddingConfig = field(
        default_factory=EmbeddingConfig
    )
    dedup: DedupThresholds = field(
        default_factory=DedupThresholds
    )
    matching: MatchingWeights = field(
        default_factory=MatchingWeights
    )
    approval: ApprovalPolicy = field(
        default_factory=ApprovalPolicy
    )

    dedup_retrieval_k: int = 10
    matching_retrieval_k: int = 10
    participant_match_threshold: float = 0.5
    max_report_characters: int = 10_000

    def __post_init__(self) -> None:
        retrieval_limits = (
            self.dedup_retrieval_k,
            self.matching_retrieval_k,
        )

        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            for value in retrieval_limits
        ):
            raise TypeError(
                "Retrieval limits must be integers"
            )

        if min(retrieval_limits) < 1:
            raise ValueError(
                "Retrieval limits must be positive"
            )

        if (
            isinstance(
                self.participant_match_threshold,
                bool,
            )
            or not isinstance(
                self.participant_match_threshold,
                (int, float),
            )
        ):
            raise TypeError(
                "participant_match_threshold must be a number"
            )

        if not 0 <= self.participant_match_threshold <= 1:
            raise ValueError(
                "participant_match_threshold must be within 0..1"
            )

        if (
            isinstance(self.max_report_characters, bool)
            or not isinstance(
                self.max_report_characters,
                int,
            )
        ):
            raise TypeError(
                "max_report_characters must be an integer"
            )

        if self.max_report_characters < 100:
            raise ValueError(
                "max_report_characters must be at least 100"
            )

        if (
            self.max_report_characters
            < self.approval.min_report_characters
        ):
            raise ValueError(
                "max_report_characters must be at least the "
                "automatic-approval minimum"
            )


class AIStage(StrEnum):
    UNDERSTANDING_PROBLEM = "understanding_problem"
    CHECKING_DUPLICATES = "checking_duplicates"

    SEARCHING_ORGANIZATION_CAPABILITIES = (
        "searching_organization_capabilities"
    )

    ROUTING_PARTICIPANT_INVITATIONS = (
        "routing_participant_invitations"
    )

    VALIDATING_REPORT = "validating_report"

    FORMATTING_PROBLEM_STATEMENT = (
        "formatting_problem_statement"
    )

    CITIZEN_REVISION_REQUIRED = (
        "citizen_revision_required"
    )

    REPORT_REJECTED = "report_rejected"

    HUMAN_REVIEW_REQUIRED = (
        "human_review_required"
    )

    CHALLENGE_READY = "challenge_ready"


ProgressCallback = Callable[
    [AIStage],
    None | Awaitable[None],
]


class AIOrchestrator:
    """
    Coordinate AI services.

    Authorization, persistence, transactions, and delivery remain backend
    responsibilities.
    """

    def __init__(
        self,
        challenge_analyzer: ChallengeAnalyzer,
        deduplication: DeduplicationService,
        matching: MatchingService,
        *,
        image_evidence_analyzer: ImageEvidenceAnalyzer | None = None,
        participant_matching: ParticipantMatchingService | None = None,
        approval_policy: ApprovalPolicy | None = None,
        max_report_characters: int = 10_000,
) -> None:
        if (
            isinstance(max_report_characters, bool)
            or not isinstance(max_report_characters, int)
        ):
            raise TypeError(
                "max_report_characters must be an integer"
            )

        if max_report_characters < 100:
            raise ValueError(
                "max_report_characters must be at least 100"
            )

        self.challenge_analyzer = challenge_analyzer
        self.deduplication = deduplication
        self.matching = matching
        self.image_evidence_analyzer = image_evidence_analyzer

        self.participant_matching = (
            participant_matching
            or ParticipantMatchingService()
        )

        self.approval_policy = (
            approval_policy
            or ApprovalPolicy()
        )

        self.max_report_characters = (
            max_report_characters
        )

    @classmethod
    async def build(
        cls,
        *,
        config: AIConfig | None = None,
        provider: AIProvider | None = None,
        fallback_provider: AIProvider | None = None,
        problems: Sequence[IncomingProblem] = (),
        organizations: Sequence[
            OrganizationCapabilityProfile
        ] = (),
        participants: Sequence[
            ParticipantProfile
        ] = (),
        embedding_backend: EmbeddingBackend | None = None,
        image_evidence_analyzer: ImageEvidenceAnalyzer | None = None,
        ) -> "AIOrchestrator":
        """
        Build an in-memory orchestrator for development/tests only.

        Production must inject persistent retrievers/services instead.
        """

        if _is_production_environment():
            raise RuntimeError(
                "AIOrchestrator.build() uses in-memory retrievers "
                "and is disabled in production. Inject persistent "
                "production services instead."
            )

        settings = config or AIConfig()

        live = (
            provider
            or _provider_from_config(
                settings.provider
            )
        )

        effective_provider = (
            FallbackProvider(
                live,
                fallback_provider,
            )
            if fallback_provider is not None
            else live
        )

        if (
            embedding_backend is None
            and live.name != "mock"
        ):
            raise ValueError(
                "A live semantic embedding_backend must be supplied "
                "when using a live AI provider. Pass a production "
                "embedding backend, or explicitly use "
                "HashingEmbeddingBackend only for a deliberate baseline."
            )

        selected_embedding_backend = (
            embedding_backend
            or HashingEmbeddingBackend(
                settings.embedding
            )
        )

        if isinstance(
            selected_embedding_backend,
            HashingEmbeddingBackend,
        ):
            logger.warning(
                "Using HashingEmbeddingBackend: deterministic "
                "development/test retrieval only; do not present these "
                "results as semantic-embedding production quality."
            )

        embeddings = EmbeddingService(
            selected_embedding_backend,
            normalize_vectors=(
                settings.embedding.normalize_vectors
            ),
        )

        problem_retriever = (
            InMemoryProblemRetriever()
        )

        if problems:
            vectors = await embeddings.embed_batch(
                [
                    problem.embedding_text()
                    for problem in problems
                ],
                metadata=[
                    {
                        "problem_id":
                            problem.id or ""
                    }
                    for problem in problems
                ],
            )

            for problem, vector in zip(
                problems,
                vectors,
                strict=True,
            ):
                problem_retriever.add(
                    problem,
                    vector,
                )

        organization_retriever = (
            InMemoryOrganizationRetriever()
        )

        if organizations:
            vectors = await embeddings.embed_batch(
                [
                    organization.embedding_text()
                    for organization
                    in organizations
                ]
            )

            for organization, vector in zip(
                organizations,
                vectors,
                strict=True,
            ):
                organization_retriever.add(
                    organization,
                    vector,
                )
        selected_image_evidence_analyzer = (
            image_evidence_analyzer
        )

        if (
            selected_image_evidence_analyzer is None
            and isinstance(
                live,
                ImageEvidenceProvider,
            )
        ):
            selected_image_evidence_analyzer = (
                ImageEvidenceAnalyzer(
                    live
                )
            )

        return cls(
            ChallengeAnalyzer(
                effective_provider,
                max_report_characters=(
                    settings.max_report_characters
                ),
            ),
            DeduplicationService(
                embeddings,
                problem_retriever,
                HybridRelationshipClassifier(
                    settings.dedup,
                    effective_provider,
                ),
                top_k=settings.dedup_retrieval_k,
            ),
            MatchingService(
                embeddings,
                organization_retriever,
                MatchingFeatureExtractor(),
                WeightedMatchScorer(
                    settings.matching
                ),
                MatchExplainer(),
                top_k=settings.matching_retrieval_k,
            ),
            image_evidence_analyzer=(
                selected_image_evidence_analyzer
            ),
            participant_matching=(
                ParticipantMatchingService(
                    list(participants),
                    minimum_score=(
                        settings.participant_match_threshold
                    ),
                )
            ),
            approval_policy=settings.approval,
            max_report_characters=(
                settings.max_report_characters
            ),
        )

    async def analyze_challenge(
        self,
        report: str,
        *,
        context: Mapping[str, object] | None = None,
        on_stage: ProgressCallback | None = None,
    ) -> ChallengeAnalysis:
        """
        Return sanitized analysis only.

        This is a lower-level analysis entry point and does not enforce
        citizen-form location requirements.
        """

        clean_report = self._clean_report(
            report
        )

        safe_context = _validate_context(
            context
        )

        await _emit(
            on_stage,
            AIStage.UNDERSTANDING_PROBLEM,
        )

        try:
            return await self.challenge_analyzer.analyze(
                clean_report,
                context=safe_context,
            )

        except Exception as exc:
            _log_safe_failure(
                "Challenge analysis failed",
                exc,
            )

            raise AIProcessingError(
                "AI analysis could not be completed safely."
            ) from exc

    async def process_citizen_report(
        self,
        report: str,
        *,
        context: Mapping[str, object] | None = None,
        images: Sequence[tuple[bytes, str]] | None = None,
        on_stage: ProgressCallback | None = None,
    ) -> ReportWorkflowResult:
        """
        Process a citizen submission.

        Locality, district and PIN code are mandatory structured form fields.
        AI-generated geography is never used as a substitute for them.
        """

        clean_report = self._clean_report(
            report
        )

        safe_context = (
            _validate_citizen_submission_context(
                context
            )
        )

        await _emit(
            on_stage,
            AIStage.UNDERSTANDING_PROBLEM,
        )

        try:
            analysis = (
                await self.challenge_analyzer.analyze(
                    clean_report,
                    context=safe_context,
                )
            )

        except Exception as exc:
            _log_safe_failure(
                "Citizen report analysis failed",
                exc,
            )

            raise AIProcessingError(
                "Citizen report analysis could not be "
                "completed safely."
            ) from exc

        await _emit(
            on_stage,
            AIStage.VALIDATING_REPORT,
        )

        decision, reasons = (
            self._decide_report(
                clean_report,
                analysis,
            )
        )

        if (
            decision is ApprovalDecision.AUTO_APPROVED
            and images is not None
        ):
            if self.image_evidence_analyzer is None:
                decision = ApprovalDecision.HUMAN_REVIEW
                reasons = [
                    "The uploaded image evidence could not be "
                    "verified automatically."
                ]
            else:
                try:
                    image_evidence = (
                        await self.image_evidence_analyzer.analyze(
                            clean_report,
                            images,
                            context=safe_context,
                        )
                    )
                except Exception as exc:
                    _log_safe_failure(
                        "Image evidence analysis failed",
                        exc,
                    )
                    decision = ApprovalDecision.HUMAN_REVIEW
                    reasons = [
                        "The uploaded image evidence could not be "
                        "verified automatically."
                    ]
                else:
                    image_blockers = _image_approval_blockers(
                        image_evidence
                    )
                    if image_blockers:
                        decision = ApprovalDecision.HUMAN_REVIEW
                        reasons = image_blockers

        routed_analysis = (
            _with_review_status(
                analysis,
                decision,
            )
        )

        if (
            decision
            is ApprovalDecision.CITIZEN_REVISION_REQUIRED
        ):
            await _emit(
                on_stage,
                AIStage.CITIZEN_REVISION_REQUIRED,
            )

            return ReportWorkflowResult(
                decision=decision,
                decision_reasons=reasons,
                analysis=routed_analysis,
            )

        if (
            decision
            is ApprovalDecision.AUTO_REJECTED
        ):
            await _emit(
                on_stage,
                AIStage.REPORT_REJECTED,
            )

            return ReportWorkflowResult(
                decision=decision,
                decision_reasons=reasons,
                analysis=routed_analysis,
            )

        if (
            decision
            is ApprovalDecision.HUMAN_REVIEW
        ):
            await _emit(
                on_stage,
                AIStage.HUMAN_REVIEW_REQUIRED,
            )

            return ReportWorkflowResult(
                decision=decision,
                decision_reasons=reasons,
                analysis=routed_analysis,
            )

        await _emit(
            on_stage,
            AIStage.FORMATTING_PROBLEM_STATEMENT,
        )

        statement = _build_problem_statement(
            clean_report,
            routed_analysis,
        )

        await _emit(
            on_stage,
            AIStage.CHALLENGE_READY,
        )

        return ReportWorkflowResult(
            decision=(
                ApprovalDecision.AUTO_APPROVED
            ),
            decision_reasons=reasons,
            analysis=routed_analysis,
            problem_statement=statement,
        )

    async def finalize_human_approved_report(
        self,
        report: str,
        analysis: ChallengeAnalysis,
        *,
        on_stage: ProgressCallback | None = None,
    ) -> ProblemStatement:
        """
        Build a problem statement after explicit human approval.

        Human approval may override the review decision, but the supplied analysis
        must still be the sanitized analysis produced for this exact citizen report.
        No new AI call occurs here.
        """

        clean_report = self._clean_report(
            report
        )

        if not isinstance(
            analysis,
            ChallengeAnalysis,
        ):
            raise TypeError(
                "analysis must be a ChallengeAnalysis"
            )

        expected_fingerprint = report_fingerprint(
            clean_report
        )

        if (
            analysis.source_report_fingerprint
            != expected_fingerprint
        ):
            raise ValueError(
                "The supplied analysis does not belong to "
                "the supplied citizen report."
            )

        confirmed_analysis = _set_review_status(
            analysis,
            ReviewStatus.HUMAN_CONFIRMED,
        )

        await _emit(
            on_stage,
            AIStage.FORMATTING_PROBLEM_STATEMENT,
        )

        statement = _build_problem_statement(
            clean_report,
            confirmed_analysis,
        )

        await _emit(
            on_stage,
            AIStage.CHALLENGE_READY,
        )

        return statement

    async def find_duplicates(
        self,
        problem: IncomingProblem,
        *,
        on_stage: ProgressCallback | None = None,
    ) -> DedupResult:
        await _emit(
            on_stage,
            AIStage.CHECKING_DUPLICATES,
        )

        return await self.deduplication.find_duplicates(
            problem
        )

    async def match_organizations(
        self,
        challenge: ChallengeRequirements,
        *,
        on_stage: ProgressCallback | None = None,
    ) -> MatchingResultSet:
        await _emit(
            on_stage,
            AIStage.SEARCHING_ORGANIZATION_CAPABILITIES,
        )

        return await self.matching.match(
            challenge
        )

    async def route_participant_invitations(
        self,
        problem: ProblemStatement,
        *,
        approved: bool = False,
        on_stage: ProgressCallback | None = None,
    ) -> ParticipantRoutingResult:
        """
        Return eligible participant IDs.

        Backend code owns approval authorization and actual notification
        delivery.
        """

        if approved is not True:
            raise PermissionError(
                "Participant routing requires an approved problem."
            )

        await _emit(
            on_stage,
            AIStage.ROUTING_PARTICIPANT_INVITATIONS,
        )

        challenge = ChallengeRequirements(
            title=problem.title,
            summary=problem.problem_summary,
            domain=problem.category.value,
            skills=problem.recommended_skills,
            required_capabilities=(
                problem.recommended_capabilities
            ),
            urgency=problem.urgency,
            constraints=problem.constraints,
        )

        return self.participant_matching.match(
            challenge
        )

    def _clean_report(
        self,
        report: str,
    ) -> str:
        if not isinstance(report, str):
            raise TypeError(
                "report must be a string"
            )

        if len(report) > self.max_report_characters:
            raise ValueError(
                "report exceeds the configured maximum length"
            )

        clean = " ".join(
            report.split()
        ).strip()

        if len(clean) < 3:
            raise ValueError(
                "report must contain at least "
                "3 non-whitespace characters"
            )

        return clean

    def _decide_report(
        self,
        report: str,
        analysis: ChallengeAnalysis,
    ) -> tuple[
        ApprovalDecision,
        list[str],
    ]:
        """Map grounded AI analysis to a deterministic workflow decision."""

        if (
            analysis.report_quality
            is ReportQuality.GIBBERISH
        ):
            if _safe_to_auto_reject(
                analysis
            ):
                return (
                    ApprovalDecision.AUTO_REJECTED,
                    [
                        "The submission is confidently classified "
                        "as unintelligible or obvious junk and "
                        "contains no grounded civic facts."
                    ],
                )

            return (
                ApprovalDecision.HUMAN_REVIEW,
                [
                    "The submission may be unusable, but the "
                    "evidence is not strong enough for safe "
                    "automatic rejection."
                ],
            )

        if (
            analysis.report_quality
            is ReportQuality.UNCERTAIN
        ):
            return (
                ApprovalDecision.HUMAN_REVIEW,
                [
                    "The report may be valid, but the AI could "
                    "not interpret it confidently enough for an "
                    "automatic decision."
                ],
            )

        if not analysis.is_sensible:
            return (
                ApprovalDecision.HUMAN_REVIEW,
                [
                    "The AI could not safely confirm that this "
                    "is a coherent civic problem."
                ],
            )

        if (
            analysis.confidence
            is ConfidenceLevel.LOW
        ):
            return (
                ApprovalDecision.HUMAN_REVIEW,
                [
                    "The AI confidence is too low to safely "
                    "request corrections or approve the report."
                ],
            )

        if (
            analysis.report_quality
            is ReportQuality.INCOMPLETE
        ):
            revision_reasons = (
                self._citizen_revision_reasons(
                    report,
                    analysis,
                    only_when_needed=True,
                )
            )

            if revision_reasons:
                return (
                    ApprovalDecision.CITIZEN_REVISION_REQUIRED,
                    revision_reasons,
                )

            return (
                ApprovalDecision.HUMAN_REVIEW,
                [
                    "The AI marked the report incomplete, but "
                    "no citizen-fixable missing information "
                    "could be identified safely."
                ],
            )

        citizen_reasons = (
            self._citizen_revision_reasons(
                report,
                analysis,
                only_when_needed=True,
            )
        )

        if citizen_reasons:
            return (
                ApprovalDecision.CITIZEN_REVISION_REQUIRED,
                citizen_reasons,
            )

        human_reasons = (
            self._automatic_approval_blockers(
                report,
                analysis,
            )
        )

        if human_reasons:
            return (
                ApprovalDecision.HUMAN_REVIEW,
                human_reasons,
            )

        return (
            ApprovalDecision.AUTO_APPROVED,
            [
                "All automatic-approval safety checks passed."
            ],
        )

    def _citizen_revision_reasons(
        self,
        report: str,
        analysis: ChallengeAnalysis,
        *,
        only_when_needed: bool = False,
    ) -> list[str]:
        reasons: list[str] = []
        policy = self.approval_policy

        if (
            len(report)
            < policy.min_report_characters
        ):
            reasons.append(
                "Please provide a little more detail about "
                "the reported problem."
            )

        for name in (
            analysis.missing_required_information
        ):
            if _is_submission_location_field(
                name
            ):
                continue

            reasons.append(
                f"Please provide: "
                f"{_humanize_field(name)}."
            )

        if (
            not analysis.is_actionable
            and not reasons
        ):
            reasons.append(
                "Please add enough specific information for "
                "a team to understand what needs attention "
                "and what is happening."
            )

        if (
            analysis.fields_needing_confirmation
            and not analysis.grounding_issues
        ):
            for name in (
                analysis.fields_needing_confirmation
            ):
                if _is_submission_location_field(
                    name
                ):
                    continue

                reasons.append(
                    f"Please confirm: "
                    f"{_humanize_field(name)}."
                )

        reasons = _limit_reasons(
            _unique(reasons)
        )

        if reasons or only_when_needed:
            return reasons

        return [
            "Please add the missing details needed to "
            "understand and act on this report."
        ]

    def _automatic_approval_blockers(
        self,
        report: str,
        analysis: ChallengeAnalysis,
    ) -> list[str]:
        reasons: list[str] = []

        if (
            analysis.report_quality
            is not ReportQuality.COMPLETE
        ):
            reasons.append(
                "The report is not classified as complete."
            )

        if not analysis.is_actionable:
            reasons.append(
                "The report is not actionable enough for "
                "automatic approval."
            )

        if (
            analysis.confidence
            is not ConfidenceLevel.HIGH
        ):
            reasons.append(
                "The AI analysis confidence is not high "
                "enough for automatic approval."
            )

        if analysis.grounding_issues:
            reasons.append(
                "The AI produced information that could "
                "not be fully grounded in the citizen report."
            )

        policy = self.approval_policy

        if (
            policy.require_location
            and not _has_usable_location(
                analysis
            )
        ):
            reasons.append(
                "The required submitted location could not "
                "be preserved safely in the AI analysis."
            )

        if policy.require_source_evidence:
            if not analysis.source_evidence:
                reasons.append(
                    "The AI analysis does not contain "
                    "grounded source evidence."
                )

            elif not _source_evidence_is_consistent(
                report,
                analysis,
            ):
                reasons.append(
                    "The AI source evidence is inconsistent "
                    "with the submitted citizen report."
                )

        if (
            policy.require_core_evidence
            and not _has_grounded_core_evidence(
                report,
                analysis,
            )
        ):
            reasons.append(
                "The report does not contain grounded "
                "evidence for a substantive civic-problem fact."
            )

        return _limit_reasons(
            _unique(reasons)
        )


async def _emit(
    callback: ProgressCallback | None,
    stage: AIStage,
) -> None:
    """Progress callbacks must never break AI processing."""

    if callback is None:
        return

    try:
        result = callback(stage)

        if inspect.isawaitable(result):
            await result

    except Exception as exc:
        _log_safe_failure(
            f"Progress callback failed for stage {stage.value}",
            exc,
        )


def _log_safe_failure(
    event: str,
    exc: BaseException,
) -> None:
    """Log only the failure category, never report/provider contents."""

    logger.error(
        "%s; exception_type=%s",
        event,
        type(exc).__name__,
    )


def _provider_from_config(
    config: ProviderConfig,
) -> AIProvider:
    if config.name == "mock":
        if _is_production_environment():
            raise RuntimeError(
                "MockProvider is not allowed in production."
            )

        return MockProvider()

    if config.name == "gemini":
        from ai.providers.gemini import GeminiProvider

        return GeminiProvider(config)

    raise ValueError(
        f"Unsupported AI provider: {config.name}"
    )


def _is_production_environment() -> bool:
    values = (
        os.getenv("APP_ENV"),
        os.getenv("ENVIRONMENT"),
        os.getenv("ENV"),
        os.getenv("PYTHON_ENV"),
        os.getenv("JANSETU_ENV"),
    )

    return any(
        value
        and value.strip().casefold()
        in {"prod", "production"}
        for value in values
    )


def _validate_context(
    context: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    """Reject malformed or unbounded context."""

    if context is None:
        return None

    if not isinstance(context, Mapping):
        raise TypeError(
            "context must be a mapping"
        )

    if len(context) > 50:
        raise ValueError(
            "context contains too many fields"
        )

    safe: dict[str, object] = {}
    normalized_keys: set[str] = set()
    total_size = 0

    for key, value in context.items():
        if (
            not isinstance(key, str)
            or not key.strip()
        ):
            raise TypeError(
                "context keys must be non-empty strings"
            )

        if not _is_json_like(value):
            raise TypeError(
                f"context field {key!r} contains "
                "an unsupported value type"
            )

        normalized_key = key.strip()
        folded = normalized_key.casefold()

        if folded in normalized_keys:
            raise ValueError(
                "duplicate normalized context key: "
                f"{normalized_key!r}"
            )

        normalized_keys.add(
            folded
        )

        total_size += (
            len(normalized_key)
            + len(str(value))
        )

        if total_size > 10_000:
            raise ValueError(
                "context is too large"
            )

        safe[normalized_key] = value

    return safe


def _validate_citizen_submission_context(
    context: Mapping[str, object] | None,
) -> Mapping[str, object]:
    """
    Enforce the citizen-report form contract.

    Locality, district and PIN are application-supplied geography.
    The AI must never infer missing values as a substitute.
    """

    safe_context = _validate_context(
        context
    )

    if safe_context is None:
        raise ValueError(
            "Citizen report context is required."
        )

    normalized = {
        key.casefold(): value
        for key, value in safe_context.items()
    }

    missing: list[str] = []

    for field_name in (
        _REQUIRED_CITIZEN_LOCATION_FIELDS
    ):
        value = normalized.get(
            field_name
        )

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            missing.append(
                field_name
            )

    if missing:
        readable = ", ".join(
            name.replace("_", " ")
            for name in missing
        )

        raise ValueError(
            "Citizen report context is missing required "
            f"location fields: {readable}."
        )

    locality = (
        str(normalized["locality"]).strip()
    )

    district = (
        str(normalized["district"]).strip()
    )

    pin_code = (
        str(normalized["pin_code"]).strip()
    )

    if not _PIN_CODE_PATTERN.fullmatch(
        pin_code
    ):
        raise ValueError(
            "pin_code must contain exactly 6 digits."
        )

    canonical = dict(
        safe_context
    )

    for key in list(canonical):
        if (
            key.casefold()
            in _REQUIRED_CITIZEN_LOCATION_FIELDS
        ):
            canonical.pop(key)

    canonical.update(
        {
            "locality": locality,
            "district": district,
            "pin_code": pin_code,
        }
    )

    return canonical


def _is_json_like(
    value: object,
    depth: int = 0,
) -> bool:
    if depth > 3:
        return False

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return True

    if isinstance(value, Mapping):
        return (
            len(value) <= 50
            and all(
                isinstance(key, str)
                and _is_json_like(
                    child,
                    depth + 1,
                )
                for key, child
                in value.items()
            )
        )

    if (
        isinstance(value, Sequence)
        and not isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        )
    ):
        return (
            len(value) <= 50
            and all(
                _is_json_like(
                    child,
                    depth + 1,
                )
                for child in value
            )
        )

    return False


def _image_approval_blockers(
    evidence: ImageEvidenceAnalysis,
) -> list[str]:
    reasons: list[str] = []

    if evidence.usable_image_count == 0:
        reasons.append(
            "The uploaded photos could not be assessed reliably."
        )

    if evidence.relevant_image_count == 0:
        reasons.append(
            "The uploaded photos do not clearly support the "
            "reported civic problem."
        )

    if evidence.status is ImageEvidenceStatus.UNRELATED:
        reasons.append(
            "The uploaded photos appear unrelated to the "
            "reported civic problem."
        )
    elif evidence.status is ImageEvidenceStatus.CONTRADICTS:
        reasons.append(
            "The uploaded photos may conflict with the central "
            "reported claim and require human review."
        )
    elif evidence.status is ImageEvidenceStatus.UNCLEAR:
        reasons.append(
            "The uploaded photos are too unclear for automatic "
            "evidence verification."
        )

    if evidence.confidence is ConfidenceLevel.LOW:
        reasons.append(
            "Image-evidence confidence is too low for automatic approval."
        )

    return _limit_reasons(
        _unique(reasons)
    )


def _safe_to_auto_reject(
    analysis: ChallengeAnalysis,
) -> bool:
    return (
        analysis.report_quality
        is ReportQuality.GIBBERISH
        and analysis.confidence
        is ConfidenceLevel.HIGH
        and analysis.is_sensible is False
        and analysis.is_actionable is False
        and not analysis.extracted_facts
        and not analysis.source_evidence
    )


def _has_usable_location(
    analysis: ChallengeAnalysis,
) -> bool:
    """
    Require the complete structured citizen location.

    Coordinates are optional and never replace locality, district or PIN.
    """

    location = analysis.geographic_context

    if location is None:
        return False

    locality = (
        location.locality.strip()
        if isinstance(
            location.locality,
            str,
        )
        else ""
    )

    district = (
        location.district.strip()
        if isinstance(
            location.district,
            str,
        )
        else ""
    )

    pin_code = (
        location.pin_code.strip()
        if isinstance(
            location.pin_code,
            str,
        )
        else ""
    )

    return bool(
        locality
        and district
        and _PIN_CODE_PATTERN.fullmatch(
            pin_code
        )
    )


def _source_evidence_is_consistent(
    report: str,
    analysis: ChallengeAnalysis,
) -> bool:
    """Validate structural consistency between evidence and grounded facts."""

    evidence_by_field: dict[
        str,
        list[str],
    ] = {}

    for evidence in analysis.source_evidence:
        field_name = (
            evidence.field
            .strip()
            .casefold()
        )

        if (
            not field_name
            or not _quote_appears_in_report(
                evidence.source_quote,
                report,
            )
        ):
            return False

        evidence_by_field.setdefault(
            field_name,
            [],
        ).append(
            evidence.source_quote
        )

    for field_name, grounded_value in (
        analysis.extracted_facts.items()
    ):
        key = field_name.casefold()

        quotes = evidence_by_field.get(
            key,
            [],
        )

        if not quotes:
            return False

        normalized_value = _normalize_text(
            grounded_value
        )

        if not any(
            normalized_value
            == _normalize_text(quote)
            for quote in quotes
        ):
            return False

    if (
        analysis.affected_population is not None
        and "affected_population"
        not in evidence_by_field
    ):
        return False

    return True


_AUXILIARY_FACT_TOKENS = frozenset(
    {
        "location",
        "locality",
        "district",
        "state",
        "pin",
        "code",
        "pincode",
        "latitude",
        "longitude",
        "coordinate",
        "coordinates",
        "date",
        "time",
        "reported",
        "report",
        "affected",
        "population",
        "count",
        "number",
        "people",
        "category",
        "domain",
        "subdomain",
        "urgency",
        "severity",
        "tag",
        "tags",
        "skill",
        "skills",
        "capability",
        "capabilities",
        "name",
        "citizen",
        "reporter",
        "email",
        "phone",
        "mobile",
        "contact",
        "address",
        "aadhaar",
        "aadhar",
        "identity",
        "identifier",
        "id",
    }
)


def _has_grounded_core_evidence(
    report: str,
    analysis: ChallengeAnalysis,
) -> bool:
    """
    Require a grounded substantive citizen fact before automatic approval.

    Geography, population, dates, classifications and contact data cannot be
    the sole evidence supporting automatic approval.
    """

    evidence_by_field: dict[
        str,
        list[str],
    ] = {}

    for evidence in analysis.source_evidence:
        field_key = (
            evidence.field
            .strip()
            .casefold()
        )

        if not field_key:
            continue

        if not _quote_appears_in_report(
            evidence.source_quote,
            report,
        ):
            continue

        evidence_by_field.setdefault(
            field_key,
            [],
        ).append(
            evidence.source_quote
        )

    for field_name, fact_value in (
        analysis.extracted_facts.items()
    ):
        if _is_auxiliary_fact_field(
            field_name
        ):
            continue

        quotes = evidence_by_field.get(
            field_name.casefold(),
            [],
        )

        if not quotes:
            continue

        normalized_fact = _normalize_text(
            fact_value
        )

        if not normalized_fact:
            continue

        for quote in quotes:
            normalized_quote = (
                _normalize_text(
                    quote
                )
            )

            if (
                normalized_quote
                != normalized_fact
            ):
                continue

            words = normalized_quote.split()

            if (
                len(words) >= 2
                and len(normalized_quote) >= 8
            ):
                return True

    return False


def _is_auxiliary_fact_field(
    field_name: str,
) -> bool:
    tokens = {
        token
        for token in re.split(
            r"[^a-zA-Z0-9]+",
            field_name.casefold(),
        )
        if token
    }

    if not tokens:
        return True

    return tokens.issubset(
        _AUXILIARY_FACT_TOKENS
    )


def _is_submission_location_field(
    value: str,
) -> bool:
    """Identify form-owned geographic fields returned by the model."""

    tokens = {
        token
        for token in re.split(
            r"[^a-zA-Z0-9]+",
            value.casefold(),
        )
        if token
    }

    if not tokens:
        return False

    if (
        "pin" in tokens
        and "code" in tokens
    ):
        return True

    return bool(
        tokens
        & {
            "location",
            "area",
            "locality",
            "district",
            "pincode",
            "latitude",
            "longitude",
            "coordinate",
            "coordinates",
        }
    )


def _quote_appears_in_report(
    quote: str,
    report: str,
) -> bool:
    normalized_quote = (
        _normalize_text(
            quote
        )
    )

    normalized_report = (
        _normalize_text(
            report
        )
    )

    return (
        bool(normalized_quote)
        and normalized_quote
        in normalized_report
    )


def _normalize_text(
    value: str,
) -> str:
    return " ".join(
        value.casefold().split()
    )


def _with_review_status(
    analysis: ChallengeAnalysis,
    decision: ApprovalDecision,
) -> ChallengeAnalysis:
    status = {
        ApprovalDecision.AUTO_APPROVED:
            ReviewStatus.AUTO_APPROVED,

        ApprovalDecision.CITIZEN_REVISION_REQUIRED:
            ReviewStatus.CITIZEN_REVISION_REQUIRED,

        ApprovalDecision.AUTO_REJECTED:
            ReviewStatus.AUTO_REJECTED,

        ApprovalDecision.HUMAN_REVIEW:
            ReviewStatus.HUMAN_REVIEW_REQUIRED,
    }[decision]

    return _set_review_status(
        analysis,
        status,
    )


def _set_review_status(
    analysis: ChallengeAnalysis,
    status: ReviewStatus,
) -> ChallengeAnalysis:
    payload = analysis.model_dump(
        mode="python"
    )

    payload["review_status"] = status

    return ChallengeAnalysis.model_validate(
        payload
    )


def _humanize_field(
    value: str,
) -> str:
    return " ".join(
        value
        .replace(".", " ")
        .replace("_", " ")
        .split()
    )


def _unique(
    values: list[str],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = " ".join(
            value.split()
        ).strip()

        if not cleaned:
            continue

        key = cleaned.casefold()

        if key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


def _limit_reasons(
    reasons: list[str],
) -> list[str]:
    return reasons[
        :_MAX_DECISION_REASONS
    ]


def _build_problem_statement(
    report: str,
    analysis: ChallengeAnalysis,
) -> ProblemStatement:
    """
    Build the internal approved-problem record without another AI call.

    Public/student consumers must use ProblemStatement.to_public().
    """

    return ProblemStatement(
        title=analysis.normalized_title,
        problem_summary=analysis.summary,
        citizen_report=report,
        category=analysis.category,
        location=analysis.geographic_context,
        urgency=analysis.urgency,
        severity=analysis.severity,
        citizen_facts=(
            analysis.extracted_facts
        ),
        source_evidence=(
            analysis.source_evidence
        ),
        constraints=analysis.constraints,
        unknowns=analysis.unknown_fields,
        recommended_capabilities=(
            analysis.required_capabilities
        ),
        recommended_skills=(
            analysis.skills
        ),
    )
