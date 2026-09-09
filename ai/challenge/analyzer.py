"""Conservative citizen-report source preservation.

Quotes establish provenance, not truth or semantic entailment. Backend-validated
context is required for structured geography. Population claims remain in source
text; the optional scalar is withheld without making the whole report uncertain.
Evidence preserves the whole report up to 500 characters; longer contexts require review.
Provider classification/skills remain derived metadata and require real-world evaluation.
This module alone does not make the application production ready.
"""
from __future__ import annotations
import re
import math
import unicodedata
import hashlib
from collections.abc import Mapping
from ai.challenge.prompts import CHALLENGE_PROMPT_VERSION, CHALLENGE_SYSTEM_PROMPT, build_challenge_prompt
from ai.providers.base import AIProvider
from ai.schemas.challenge import ChallengeAnalysis, ConfidenceLevel, GeographicContext, ProblemCategory, ReportQuality, ReviewStatus, Severity, SourceEvidence, Urgency
_ALLOWED_CONTEXT_FIELDS = frozenset({'title', 'locality', 'district', 'state', 'pin_code', 'latitude', 'longitude'})
_LOCATION_FIELDS = ('locality', 'district', 'state', 'pin_code', 'latitude', 'longitude')
_MAX_EVIDENCE_SPAN_LENGTH = 500
_MAX_SUMMARY_LENGTH = 1000

_PRESENTATION_REQUIREMENTS = """
Presentation-field requirements:
- normalized_title must be a concise, polished, professional English title for the civic problem.
- summary must be a clear, polished, professional English description of the problem, normally 1-3 sentences.
- If the citizen report is Hindi, Hinglish, transliterated Hindi, another supported language, broken English,
  informal, or poorly worded, translate/paraphrase it into natural English for normalized_title and summary.
- normalized_title and summary may improve wording, grammar, and readability, but must preserve the original
  meaning and must not introduce any factual claim, location, number, cause, consequence, duration, severity,
  or other detail that is absent from citizen_report or trusted_context.
- Remove conversational request wording such as "please fix", "help us", or "kindly solve" when it does not
  describe the problem.
- These rules apply only to normalized_title and summary. source_evidence and grounded citizen facts must
  remain source-preserving according to the existing schema and grounding rules.
""".strip()

_NON_ACTIONABLE_REPORT_PATTERNS = (
    re.compile(
        r"\b(?:there is|there's|there are)\s+no\s+"
        r"(?:civic\s+)?(?:issue|problem)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:issue|problem)\s+(?:is|was|has been)\s+"
        r"(?:fully\s+)?(?:resolved|fixed|repaired)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:already|now)\s+"
        r"(?:fixed|resolved|repaired)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:is|are)\s+(?:now\s+)?working\s+"
        r"(?:properly|correctly|fine)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:is|are)\s+not\s+broken\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\broad\s+is\s+(?:now\s+)?fine\b",
        re.IGNORECASE,
    ),
)

_INSTRUCTION_LIKE_REPORT_PATTERNS = (
    re.compile(
        r"\bignore\s+(?:all\s+|the\s+)?"
        r"(?:previous|prior|system|developer)\s+instructions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:approve|auto[- ]?approve|reject|auto[- ]?reject)\s+"
        r"(?:this|the)\s+report\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:classify|mark|set|change)\s+.{0,40}\b"
        r"(?:category|severity|urgency|confidence|report_quality)\b",
        re.IGNORECASE,
    ),
)


_HIGH_IMPACT_REPORT_PATTERNS = (
    re.compile(
        r"\b(?:unsafe|contaminated|toxic|poison(?:ed|ing)?|"
        r"fire|smoke|sparking|electrocution|electric\s+shock|"
        r"injur(?:y|ies|ed)|accident(?:s)?|collaps(?:e|ed|ing)|"
        r"flood(?:ed|ing)?|outbreak|hospitali[sz]ed|dangerous)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdirty\s+(?:drinking\s+)?water\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\blive\s+(?:electrical\s+)?wire\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no\s+drinking\s+water|no\s+electricity|"
        r"power\s+outage|blocked\s+road)\b",
        re.IGNORECASE,
    ),
)

_CRITICAL_IMPACT_REPORT_PATTERNS = (
    re.compile(
        r"\b(?:death|dead|fatal|fatality|life[- ]threatening|"
        r"electrocuted|electrocution|explosion|major\s+fire|"
        r"severe\s+injur(?:y|ies)|immediate\s+danger)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:building|bridge|wall)\s+(?:has\s+)?collapsed\b",
        re.IGNORECASE,
    ),
)

_SPECIALIST_RECOMMENDATION_TOKEN_GROUPS = (
    frozenset(
        {
            "nuclear",
            "radiation",
            "radiological",
        }
    ),
    frozenset(
        {
            "explosive",
            "explosives",
            "bomb",
            "ordnance",
        }
    ),
    frozenset(
        {
            "biohazard",
            "biological",
        }
    ),
)

class ChallengeAnalyzer:

    def __init__(self, provider: AIProvider, *, max_report_characters: int=10000) -> None:
        if isinstance(max_report_characters, bool) or not isinstance(max_report_characters, int):
            raise TypeError('max_report_characters must be an integer')
        if max_report_characters < 100:
            raise ValueError('max_report_characters must be at least 100')
        self.provider = provider
        self.max_report_characters = max_report_characters

    async def analyze(self, raw_report: str, *, context: Mapping[str, object] | None=None) -> ChallengeAnalysis:
        if not isinstance(raw_report, str):
            raise TypeError('raw_report must be a string')
        if len(raw_report) > self.max_report_characters:
            raise ValueError(f'Citizen report must not exceed {self.max_report_characters:,} characters')
        if sum((not char.isspace() for char in raw_report)) < 3:
            raise ValueError('Citizen report must contain at least 3 non-whitespace characters')
        normalized = _clean_whitespace(raw_report)
        allowed_context = _sanitize_context(context)
        challenge_prompt = (
            build_challenge_prompt(normalized, allowed_context)
            + "\n\n"
            + _PRESENTATION_REQUIREMENTS
        )
        analysis = await self.provider.generate_structured(
            challenge_prompt,
            ChallengeAnalysis,
            system_prompt=CHALLENGE_SYSTEM_PROMPT,
            temperature=0.0,
        )
        if isinstance(analysis, ChallengeAnalysis):
            analysis = analysis.model_dump(mode='python')
        analysis = ChallengeAnalysis.model_validate(analysis)
        return _ground_analysis(analysis, normalized, allowed_context)

def _ground_analysis(
    analysis: ChallengeAnalysis,
    raw_report: str,
    context: Mapping[str, str | int | float],
) -> ChallengeAnalysis:
    confirmation = list(
        analysis.fields_needing_confirmation
    )

    grounding_issues = list(
        analysis.grounding_issues
    )

    valid_evidence = _ground_source_evidence(
        analysis.source_evidence,
        raw_report,
        confirmation,
        grounding_issues,
    )

    evidence_by_field = _index_evidence(
        valid_evidence
    )

    grounded_facts = _ground_extracted_facts(
        analysis,
        evidence_by_field,
        confirmation,
        grounding_issues,
    )

    grounded_constraints = _ground_constraints(
        analysis.constraints,
        raw_report,
        confirmation,
        grounding_issues,
    )

    # Population is an optional model-derived scalar.
    #
    # If the citizen actually supplied a population claim and that claim survived
    # source grounding, preserve the claim through evidence/facts but withhold the
    # numeric scalar without downgrading the whole report.
    #
    # If the model supplied a population number with no grounded population
    # evidence, treat it as an unsupported model-generated claim.
    affected_population = None

    if analysis.affected_population is not None:
        population_evidence = (
            evidence_by_field.get(
                "affected_population",
                [],
            )
        )

        if not population_evidence:
            grounding_issues.append(
                "Affected population is unsupported by "
                "grounded citizen-report evidence"
            )

            confirmation.append(
                "affected_population"
            )

    geography = _ground_geography(
        context
    )

    supplied_title = context.get(
        "title"
    )

    # Title is a presentation field: allow a faithful English rewrite from
    # the provider. Raw citizen wording remains preserved in evidence/facts.
    title = _clean_whitespace(
        analysis.normalized_title
    )

    if not title:
        title = _build_grounded_title(
            raw_report,
            supplied_title,
        )

    report_quality = _normalize_report_quality(
        analysis
    )

    semantic_issue = _explicit_non_actionable_reason(
        raw_report
    )

    effective_actionable = (
        analysis.is_actionable
        and semantic_issue is None
    )

    effective_urgency, effective_severity = (
        _sanitize_priority_metadata(
            analysis,
            raw_report,
        )
    )

    effective_capabilities = _sanitize_recommendations(
        analysis.required_capabilities,
        raw_report,
    )

    effective_skills = _sanitize_recommendations(
        analysis.skills,
        raw_report,
    )

    if semantic_issue is not None:
        grounding_issues.append(
            semantic_issue
        )

        confirmation.append(
            "is_actionable"
        )

    priority_source_spans = [
        evidence.source_quote
        for evidence in valid_evidence
    ]

    priority_source_spans.extend(
        grounded_constraints
    )

    # Summary is also a presentation field. It may translate/paraphrase the
    # citizen report into professional English, but the provider is explicitly
    # forbidden above from adding unsupported facts.
    summary = _clean_whitespace(
        analysis.summary
    )

    if not summary:
        summary, _ = _build_grounded_summary(
            raw_report,
            priority_source_spans=(
                priority_source_spans
            ),
        )

    # IMPORTANT:
    # Detect a contradiction BEFORE any gibberish cleanup.
    #
    # If the provider labels a report as gibberish while deterministic
    # grounding has successfully preserved source-backed evidence,
    # facts, or constraints, automatic rejection is unsafe.
    gibberish_conflicts_with_grounded_content = (
        report_quality
        is ReportQuality.GIBBERISH
        and bool(
            valid_evidence
            or grounded_facts
            or grounded_constraints
        )
    )

    if gibberish_conflicts_with_grounded_content:
        grounding_issues.append(
            "Gibberish classification conflicts with "
            "grounded citizen-report content"
        )
        confirmation.append(
            "report_quality"
        )

    updates: dict[str, object] = {
        "prompt_version":
            CHALLENGE_PROMPT_VERSION,

        "source_report_fingerprint":
            report_fingerprint(raw_report),

        "review_status":
            ReviewStatus.NEEDS_REVIEW,

        "report_quality":
            report_quality,

        "is_actionable":
            effective_actionable,

        "urgency":
            effective_urgency,

        "severity":
            effective_severity,

        "required_capabilities":
            effective_capabilities,

        "skills":
            effective_skills,

        "normalized_title":
            title,

        "summary":
            summary,

        "affected_population":
            affected_population,

        "geographic_context":
            geography,

        "extracted_facts":
            grounded_facts,

        "source_evidence":
            valid_evidence,

        "constraints":
            grounded_constraints,

        "fields_needing_confirmation":
            _unique(confirmation),

        "grounding_issues":
            _unique(grounding_issues),
    }

    if (
        report_quality
        is ReportQuality.GIBBERISH
    ):
        updates.update(
            {
                "category":
                    ProblemCategory.OTHERS,

                "domain":
                    None,

                "subdomain":
                    None,

                "urgency":
                    Urgency.UNKNOWN,

                "severity":
                    Severity.UNKNOWN,

                "affected_population":
                    None,

                "required_capabilities":
                    [],

                "skills":
                    [],

                "tags":
                    [],

                "is_sensible":
                    False,

                "is_actionable":
                    False,
            }
        )

        # Only erase grounded material when nothing meaningful was
        # deterministically grounded.
        #
        # If grounded content conflicts with the gibberish label, retain it
        # so downstream rejection safeguards and human reviewers can see it.
        if not gibberish_conflicts_with_grounded_content:
            updates.update(
                {
                    "constraints":
                        [],

                    "extracted_facts":
                        {},

                    "source_evidence":
                        [],
                }
            )

    # Any grounding contradiction means automatic interpretation is unsafe.
    # Fail closed to UNCERTAIN / LOW rather than trusting the model.
    if grounding_issues:
        updates["report_quality"] = (
            ReportQuality.UNCERTAIN
        )

        updates["confidence"] = (
            ConfidenceLevel.LOW
        )

    updates[
        "fields_needing_confirmation"
    ] = _bounded_messages(
        confirmation,
        20,
    )

    updates[
        "grounding_issues"
    ] = _bounded_messages(
        grounding_issues,
        30,
    )

    payload = analysis.model_dump(
        mode="python"
    )

    payload.update(
        updates
    )

    return ChallengeAnalysis.model_validate(
        payload
    )

def _ground_source_evidence(proposed_evidence: list[SourceEvidence], raw_report: str, confirmation: list[str], grounding_issues: list[str]) -> list[SourceEvidence]:
    grounded: list[SourceEvidence] = []
    seen: set[tuple[str, str]] = set()
    for evidence in proposed_evidence:
        source_span = _find_unique_containing_source_span(evidence.source_quote, raw_report)
        if source_span is None:
            grounding_issues.append(f'Evidence quote is absent, ambiguous, or cannot be preserved with full context: {evidence.field}')
            confirmation.append(evidence.field)
            continue
        key = (evidence.field.casefold(), source_span)
        if key in seen:
            continue
        seen.add(key)
        grounded.append(SourceEvidence(field=evidence.field, source_quote=source_span))
    return grounded

def _index_evidence(evidence: list[SourceEvidence]) -> dict[str, list[str]]:
    indexed: dict[str, list[str]] = {}
    for item in evidence:
        key = item.field.casefold()
        indexed.setdefault(key, []).append(item.source_quote)
    return indexed

def _ground_extracted_facts(analysis: ChallengeAnalysis, evidence_by_field: Mapping[str, list[str]], confirmation: list[str], grounding_issues: list[str]) -> dict[str, str]:
    grounded: dict[str, str] = {}
    for field_name in analysis.extracted_facts:
        spans = _unique_source_spans(evidence_by_field.get(field_name.casefold(), []))
        if not spans:
            grounding_issues.append(f'Extracted fact lacks authoritative source context: {field_name}')
            confirmation.append(field_name)
            continue
        if len(spans) > 1:
            grounding_issues.append(f'Extracted fact has multiple distinct source contexts and cannot be resolved safely: {field_name}')
            confirmation.append(field_name)
            continue
        grounded[field_name] = spans[0]
    return grounded

def _ground_constraints(
    constraints: list[str],
    raw_report: str,
    confirmation: list[str],
    grounding_issues: list[str],
) -> list[str]:
    grounded: list[str] = []

    for constraint in constraints:
        source_span = _find_unique_containing_source_span(
            constraint,
            raw_report,
            max_length=200,
        )

        if source_span is None:
            grounding_issues.append(
                "Unsupported, ambiguous, or oversized constraint"
            )
            confirmation.append(
                "constraints"
            )
            continue

        grounded.append(
            source_span
        )

    return _unique(
        grounded
    )

def _find_unique_containing_source_span(
    fragment: str,
    source: str,
    *,
    max_length: int = _MAX_EVIDENCE_SPAN_LENGTH,
) -> str | None:
    phrase = _clean_whitespace(
        fragment
    )

    cleaned = _clean_whitespace(
        source
    )

    if (
        not phrase
        or len(phrase) > max_length
    ):
        return None

    matches = _normalized_phrase_matches(
        phrase,
        cleaned,
    )

    if len(matches) != 1:
        return None

    if len(cleaned) <= max_length:
        return cleaned

    start, end = matches[0]

    return _bounded_source_span(
        cleaned,
        start,
        end,
        max_length,
    )


def _normalized_phrase_matches(
    phrase: str,
    source: str,
) -> list[tuple[int, int]]:
    phrase = _clean_whitespace(
        phrase
    )

    source = _clean_whitespace(
        source
    )

    if not phrase:
        return []

    def word(
        char: str,
    ) -> bool:
        return (
            char == "_"
            or unicodedata.category(char)[0]
            in "LNM"
        )

    matches: list[
        tuple[int, int]
    ] = []

    start = 0

    while (
        index := source.find(
            phrase,
            start,
        )
    ) >= 0:
        end = index + len(phrase)

        left = (
            index == 0
            or not (
                word(source[index - 1])
                and word(phrase[0])
            )
        )

        right = (
            end == len(source)
            or not (
                word(source[end])
                and word(phrase[-1])
            )
        )

        if left and right:
            matches.append(
                (
                    index,
                    end,
                )
            )

        start = index + 1

    return matches


def _normalized_phrase_occurs(
    phrase: str,
    source: str,
) -> bool:
    return bool(
        _normalized_phrase_matches(
            phrase,
            source,
        )
    )


def _bounded_source_span(
    source: str,
    start: int,
    end: int,
    limit: int,
) -> str | None:
    """
    Preserve the complete sentence-like source context containing the match.

    Never manufacture a safe-looking span by cutting arbitrary characters from
    a longer sentence. If the containing context itself exceeds the field limit,
    abstain and require review.
    """

    if (
        start < 0
        or end < start
        or end > len(source)
    ):
        return None

    sentence_boundaries = ".!?"

    left_boundary = -1

    for marker in sentence_boundaries:
        left_boundary = max(
            left_boundary,
            source.rfind(
                marker,
                0,
                start,
            ),
        )

    span_start = (
        left_boundary + 1
        if left_boundary >= 0
        else 0
    )

    right_candidates = [
        index
        for marker in sentence_boundaries
        if (
            index := source.find(
                marker,
                end,
            )
        ) >= 0
    ]

    span_end = (
        min(right_candidates) + 1
        if right_candidates
        else len(source)
    )

    span = source[
        span_start:span_end
    ].strip()

    if (
        not span
        or len(span) > limit
    ):
        return None

    return span

def _unique_source_spans(spans: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for span in spans:
        cleaned = _clean_whitespace(span)
        if not cleaned:
            continue
        key = cleaned
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result

def _build_grounded_summary(
    raw_report: str,
    *,
    priority_source_spans: list[str],
) -> tuple[str, bool]:
    report = _clean_whitespace(
        raw_report
    )

    if len(report) <= _MAX_SUMMARY_LENGTH:
        return (
            report,
            False,
        )

    candidates = [
        span
        for span
        in _unique_source_spans(
            priority_source_spans
        )
        if (
            3 <= len(span)
            <= _MAX_SUMMARY_LENGTH
            and _normalized_phrase_occurs(
                span,
                report,
            )
        )
    ]

    if candidates:
        return (
            max(
                candidates,
                key=len,
            ),
            False,
        )

    return (
        "The original citizen report requires "
        "human summarization to preserve its full context.",
        True,
    )

def _ground_geography(
    context: Mapping[str, str | int | float],
) -> GeographicContext | None:
    """Build geography only from validated structured submission context."""

    grounded = {
        field_name: context[field_name]
        for field_name in _LOCATION_FIELDS
        if field_name in context
    }

    if not grounded:
        return None

    return GeographicContext.model_validate(
        grounded
    )

def _build_grounded_title(raw_report: str, supplied_title: object | None) -> str:
    if isinstance(supplied_title, str) and 3 <= len(supplied_title) <= 180:
        return supplied_title
    if 3 <= len(raw_report) <= 180:
        return raw_report
    return 'Citizen report requiring assessment'

def _matches_any_pattern(
    value: str,
    patterns: tuple[re.Pattern[str], ...],
) -> bool:
    return any(
        pattern.search(value)
        for pattern in patterns
    )


def _sanitize_priority_metadata(
    analysis: ChallengeAnalysis,
    raw_report: str,
) -> tuple[Urgency, Severity]:
    """
    Bound model-derived priority to what the report can safely support.

    This does not claim to calculate true urgency or severity. It only prevents
    unsupported high/severe escalation from surviving unchanged.
    """

    report = _clean_whitespace(
        raw_report
    )

    has_high_impact_support = _matches_any_pattern(
        report,
        _HIGH_IMPACT_REPORT_PATTERNS,
    )

    has_critical_support = _matches_any_pattern(
        report,
        _CRITICAL_IMPACT_REPORT_PATTERNS,
    )

    urgency = analysis.urgency
    severity = analysis.severity

    if (
        urgency is Urgency.CRITICAL
        and not has_critical_support
    ):
        urgency = (
            Urgency.HIGH
            if has_high_impact_support
            else Urgency.MEDIUM
        )
    elif (
        urgency is Urgency.HIGH
        and not has_high_impact_support
    ):
        urgency = Urgency.MEDIUM

    if (
        severity is Severity.SEVERE
        and not has_critical_support
    ):
        severity = (
            Severity.HIGH
            if has_high_impact_support
            else Severity.MODERATE
        )
    elif (
        severity is Severity.HIGH
        and not has_high_impact_support
    ):
        severity = Severity.MODERATE

    return urgency, severity


def _metadata_tokens(
    value: str,
) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(
            r"[^\W_]+",
            value,
            flags=re.UNICODE,
        )
    }


def _recommendation_is_supported(
    recommendation: str,
    report_tokens: set[str],
) -> bool:
    recommendation_tokens = _metadata_tokens(
        recommendation
    )

    for token_group in (
        _SPECIALIST_RECOMMENDATION_TOKEN_GROUPS
    ):
        if (
            recommendation_tokens & token_group
            and not report_tokens & token_group
        ):
            return False

    return True


def _sanitize_recommendations(
    recommendations: list[str],
    raw_report: str,
) -> list[str]:
    """
    Remove only clearly unsupported high-specialty recommendations.

    Ordinary derived recommendations remain advisory. Broad semantic relevance
    still requires evaluation rather than pretending that lexical rules prove it.
    """

    report_tokens = _metadata_tokens(
        raw_report
    )

    return [
        recommendation
        for recommendation in recommendations
        if _recommendation_is_supported(
            recommendation,
            report_tokens,
        )
    ]


def _explicit_non_actionable_reason(
    raw_report: str,
) -> str | None:
    """
    Detect only strong explicit contradictions to an active civic-problem claim.

    This is a conservative safety backstop, not a general language classifier.
    Ambiguous cases are intentionally left to the provider and human review.
    """

    report = _clean_whitespace(
        raw_report
    )

    if any(
        pattern.search(report)
        for pattern
        in _INSTRUCTION_LIKE_REPORT_PATTERNS
    ):
        return (
            "Instruction-like citizen-report text cannot "
            "establish an actionable civic problem"
        )

    if any(
        pattern.search(report)
        for pattern
        in _NON_ACTIONABLE_REPORT_PATTERNS
    ):
        return (
            "Provider actionability conflicts with explicit "
            "resolved or non-problem wording in the citizen report"
        )

    return None

def _normalize_report_quality(analysis: ChallengeAnalysis) -> ReportQuality:
    quality = analysis.report_quality
    if quality is ReportQuality.GIBBERISH and (analysis.is_sensible or analysis.is_actionable):
        return ReportQuality.UNCERTAIN
    if quality is ReportQuality.COMPLETE:
        if not analysis.is_sensible:
            return ReportQuality.UNCERTAIN
        if analysis.missing_required_information or not analysis.is_actionable:
            return ReportQuality.INCOMPLETE
    if quality is ReportQuality.INCOMPLETE and (not analysis.is_sensible):
        return ReportQuality.UNCERTAIN
    return quality

def _sanitize_context(context: Mapping[str, object] | None) -> dict[str, str | int | float]:
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise TypeError('context must be a mapping or None')
    sanitized: dict[str, str | int | float] = {}
    seen: set[str] = set()
    for raw_key, raw_value in context.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip().casefold()
        if key not in _ALLOWED_CONTEXT_FIELDS:
            continue
        if key in seen:
            raise ValueError(f'Duplicate context field: {key}')
        seen.add(key)
        if raw_value is None:
            continue
        if key in ('latitude', 'longitude'):
            if isinstance(raw_value, bool) or not isinstance(raw_value, (str, int, float)):
                raise ValueError(f'Invalid context field: {key}')
            try:
                numeric = float(raw_value)
            except (ValueError, OverflowError):
                raise ValueError(f'Invalid context field: {key}') from None
            limit = 90 if key == 'latitude' else 180
            if not math.isfinite(numeric) or not -limit <= numeric <= limit:
                raise ValueError(f'Invalid context field: {key}')
            sanitized[key] = numeric
            continue
        if key == 'pin_code' and isinstance(raw_value, int) and (not isinstance(raw_value, bool)):
            raw_value = str(raw_value)
        if not isinstance(raw_value, str):
            raise ValueError(f'Invalid context field: {key}')
        value = _clean_whitespace(raw_value)
        if not value:
            continue
        limit = 180 if key == 'title' else 200
        if len(value) > limit or (key == 'title' and len(value) < 3):
            raise ValueError(f'Invalid context field length: {key}')
        if key == 'pin_code' and (not re.fullmatch('[0-9]{6}', value)):
            raise ValueError('Invalid context field: pin_code')
        sanitized[key] = value
    return sanitized

def _clean_whitespace(value: str) -> str:
    return ' '.join(value.split()).strip()

def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_whitespace(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result

def _bounded_messages(values: list[str], limit: int) -> list[str]:
    result = _unique([_clean_whitespace(value)[:200] for value in values])
    if len(result) > limit:
        return result[:limit - 1] + ['Additional findings omitted; review the original report.']
    return result

def report_fingerprint(
    report: str,
) -> str:
    """
    Return the deterministic identity fingerprint of a normalized report.

    This binds a sanitized ChallengeAnalysis to the citizen report from which
    it was produced. It is an integrity check, not an authorization mechanism.
    """

    normalized = _clean_whitespace(
        report
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()