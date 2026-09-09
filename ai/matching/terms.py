from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from ai.embeddings.service import normalize_text

# Conservative aliases only. Broad semantic relations should be handled by
# embedding retrieval, not by pretending two different skills are identical.
_ALIAS_GROUPS = (
    ("ai", "artificial intelligence"),
    ("ml", "machine learning"),
    ("gis", "geographic information systems", "geographical information systems"),
    ("ui ux", "user interface user experience", "user interface and user experience"),
    ("iot", "internet of things"),
)

_ALIAS_MAP: dict[str, str] = {}
for group in _ALIAS_GROUPS:
    canonical = normalize_text(group[0])
    for value in group:
        normalized = normalize_text(value)
        if normalized:
            _ALIAS_MAP[normalized] = canonical


@dataclass(frozen=True, slots=True)
class RequirementMatch:
    required: str
    offered: str
    similarity: float


def normalized_term(value: str) -> str:
    normalized = normalize_text(value)
    return _ALIAS_MAP.get(normalized, normalized)


def unique_terms(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            continue
        key = normalized_term(cleaned)
        if key and key not in result:
            result[key] = cleaned
    return result


def term_similarity(left: str, right: str) -> float:
    a = normalized_term(left)
    b = normalized_term(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    jaccard = intersection / union
    containment = intersection / min(len(a_tokens), len(b_tokens))
    sequence = SequenceMatcher(None, a, b).ratio()

    # Containment is useful for pairs such as "water management" and
    # "water resource management", but it should not turn a generic
    # one-word overlap into an automatic perfect match.
    containment_score = 0.0
    if containment == 1.0:
        shorter = min(a_tokens, b_tokens, key=len)
        if len(shorter) >= 2:
            containment_score = 0.9
        elif any(len(token) >= 8 for token in shorter):
            containment_score = 0.78

    return max(jaccard, containment_score, sequence * 0.85)


def match_requirements(
    required: list[str],
    offered: list[str],
    *,
    threshold: float = 0.72,
) -> tuple[float, list[RequirementMatch], list[str]]:
    """Match requirements one-to-one so one offered skill cannot be double-counted."""

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a number")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be within 0..1")

    need = list(unique_terms(required).values())
    have = list(unique_terms(offered).values())
    if not need:
        return 0.0, [], []
    if not have:
        return 0.0, [], need

    candidates: list[tuple[float, int, int]] = []
    for required_index, required_term in enumerate(need):
        for offered_index, offered_term in enumerate(have):
            score = term_similarity(required_term, offered_term)
            if score >= threshold:
                candidates.append((score, required_index, offered_index))

    # Deterministic maximum-first greedy assignment. This is intentionally
    # conservative: one profile term may support at most one requirement.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_required: set[int] = set()
    used_offered: set[int] = set()
    selected: dict[int, RequirementMatch] = {}

    for score, required_index, offered_index in candidates:
        if required_index in used_required or offered_index in used_offered:
            continue
        used_required.add(required_index)
        used_offered.add(offered_index)
        selected[required_index] = RequirementMatch(
            required=need[required_index],
            offered=have[offered_index],
            similarity=score,
        )

    matches = [selected[index] for index in sorted(selected)]
    missing = [
        required_term
        for index, required_term in enumerate(need)
        if index not in selected
    ]
    similarity_total = sum(item.similarity for item in matches)
    coverage = similarity_total / len(need)
    return min(1.0, max(0.0, coverage)), matches, missing
