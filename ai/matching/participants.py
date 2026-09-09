from __future__ import annotations

from ai.matching.terms import match_requirements, term_similarity
from ai.schemas.matching import ChallengeRequirements
from ai.schemas.participants import (
    ParticipantMatch,
    ParticipantProfile,
    ParticipantRole,
    ParticipantRoutingResult,
)


class ParticipantMatchingService:
    """Deterministically select verified people who may receive invitations."""

    def __init__(
        self,
        participants: list[ParticipantProfile] | None = None,
        *,
        minimum_score: float = 0.5,
    ) -> None:
        if isinstance(minimum_score, bool) or not isinstance(minimum_score, (int, float)):
            raise TypeError("minimum_score must be a number")
        if not 0 <= minimum_score <= 1:
            raise ValueError("minimum_score must be within 0..1")

        self.participants = list(participants or [])
        participant_ids = [participant.id for participant in self.participants]
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("Participant IDs must be unique")
        self.minimum_score = float(minimum_score)

    def match(self, challenge: ChallengeRequirements) -> ParticipantRoutingResult:
        required = [*challenge.skills, *challenge.required_capabilities]
        if not required:
            return ParticipantRoutingResult()

        student_matches: list[ParticipantMatch] = []
        mentor_matches: list[ParticipantMatch] = []

        for participant in self.participants:
            if not (
                participant.verified
                and participant.available
                and participant.notification_opt_in
            ):
                continue

            offered = [*participant.skills, *participant.capabilities]
            expertise_score, matched, _ = match_requirements(required, offered)
            if not matched:
                continue

            domain_score = _domain_score(challenge.domain, participant.domains)
            score = _participant_score(participant.role, expertise_score, domain_score)
            if score < self.minimum_score:
                continue

            match = ParticipantMatch(
                participant_id=participant.id,
                role=participant.role,
                score=round(score, 6),
                matched_requirements=[item.required for item in matched],
            )

            if participant.role is ParticipantRole.INDUSTRY_MENTOR:
                mentor_matches.append(match)
            else:
                student_matches.append(match)

        sort_key = lambda item: (-item.score, item.participant_id)
        student_matches.sort(key=sort_key)
        mentor_matches.sort(key=sort_key)
        return ParticipantRoutingResult(
            student_matches=student_matches,
            mentor_matches=mentor_matches,
        )


def _participant_score(
    role: ParticipantRole,
    expertise_score: float,
    domain_score: float,
) -> float:
    # Domain acts as a bonus instead of a penalty. A participant who satisfies
    # half the explicit requirements should not fall below 0.5 merely because
    # their profile omitted a domain label.
    domain_bonus = 0.30 if role is ParticipantRole.INDUSTRY_MENTOR else 0.15
    return min(1.0, expertise_score + domain_bonus * domain_score * (1.0 - expertise_score))


def _domain_score(required_domain: str | None, offered_domains: list[str]) -> float:
    if not required_domain or not offered_domains:
        return 0.0
    return max(term_similarity(required_domain, offered) for offered in offered_domains)
