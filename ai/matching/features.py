from __future__ import annotations

import math
from dataclasses import dataclass, field

from ai.matching.terms import match_requirements, term_similarity, unique_terms
from ai.schemas.matching import ChallengeRequirements, OrganizationCapabilityProfile


@dataclass(frozen=True, slots=True)
class RawFeature:
    value: float
    positive_evidence: tuple[str, ...] = field(default_factory=tuple)
    gaps: tuple[str, ...] = field(default_factory=tuple)
    available: bool = True
    data_present: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.value <= 1:
            raise ValueError("RawFeature.value must be within 0..1")
        if not isinstance(self.available, bool) or not isinstance(self.data_present, bool):
            raise TypeError("RawFeature availability flags must be booleans")
        if not self.available and self.data_present:
            raise ValueError("A non-applicable feature cannot claim data_present=True")


@dataclass(frozen=True, slots=True)
class ExtractedFeatures:
    values: dict[str, RawFeature]
    hard_constraint_failures: tuple[str, ...] = field(default_factory=tuple)


def _overlap(required: list[str], offered: list[str], *, label: str) -> RawFeature:
    if not unique_terms(required):
        return RawFeature(0.0, available=False, data_present=False)

    score, matches, missing = match_requirements(required, offered)
    return RawFeature(
        score,
        tuple(
            f"Matched {label}: {item.required}"
            + (f" via {item.offered}" if item.required.casefold() != item.offered.casefold() else "")
            for item in matches
        ),
        tuple(f"Missing {label}: {item}" for item in missing),
        available=True,
    )


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius = 6371.0
    lat1, lat2 = math.radians(a_lat), math.radians(b_lat)
    dlat = lat2 - lat1
    dlon = math.radians(b_lon - a_lon)
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    value = min(1.0, max(0.0, value))
    return 2 * radius * math.asin(math.sqrt(value))


class MatchingFeatureExtractor:
    def extract(
        self,
        challenge: ChallengeRequirements,
        organization: OrganizationCapabilityProfile,
        semantic_similarity: float,
    ) -> ExtractedFeatures:
        skills = _overlap(challenge.skills, organization.skills, label="skill")
        domain = _overlap(
            [challenge.domain] if challenge.domain else [],
            organization.domains,
            label="domain",
        )
        research = _overlap(
            challenge.required_capabilities,
            organization.research_areas + organization.facilities + organization.skills,
            label="capability",
        )
        industry = _overlap(
            challenge.preferred_industry_capabilities,
            organization.industry_capabilities,
            label="industry capability",
        )

        if organization.available_capacity is None:
            capacity = RawFeature(
                0.0,
                gaps=("Available team capacity was not provided.",),
                available=True,
                data_present=False,
            )
        else:
            capacity = RawFeature(
                organization.available_capacity,
                (
                    f"Available capacity: {organization.available_capacity:.0%}",
                ) if organization.available_capacity > 0 else (),
                ("No current team capacity reported.",)
                if organization.available_capacity == 0 else (),
                available=True,
            )

        geography, distance = self._geography(challenge, organization)

        project_terms = [
            value
            for item in organization.previous_projects
            if item.verified
            for value in item.domains + item.skills
        ]
        experience = _overlap(
            ([challenge.domain] if challenge.domain else []) + challenge.skills,
            project_terms,
            label="verified prior-project evidence",
        )

        failures: list[str] = []
        if not organization.verified:
            failures.append("Organization is not verified")

        for required_facility in unique_terms(challenge.required_facilities).values():
            best = max(
                (term_similarity(required_facility, offered) for offered in organization.facilities),
                default=0.0,
            )
            if best < 0.78:
                failures.append(f"Required facility unavailable: {required_facility}")

        if (
            challenge.allowed_organization_types
            and organization.organization_type not in challenge.allowed_organization_types
        ):
            failures.append(
                f"Organization type {organization.organization_type.value} is not allowed"
            )

        if challenge.max_distance_km is not None:
            if distance is None:
                failures.append("Organization location unavailable for required distance check")
            elif distance > challenge.max_distance_km:
                failures.append(
                    f"Distance {distance:.1f} km exceeds maximum "
                    f"{challenge.max_distance_km:.1f} km"
                )

        semantic_value = max(0.0, min(1.0, semantic_similarity))
        semantic = RawFeature(
            semantic_value,
            (f"Semantic capability similarity: {semantic_value:.0%}",),
        )

        return ExtractedFeatures(
            values={
                "semantic": semantic,
                "skills": skills,
                "domain": domain,
                "research": research,
                "capacity": capacity,
                "geography": geography,
                "experience": experience,
                "industry": industry,
            },
            hard_constraint_failures=tuple(dict.fromkeys(failures)),
        )

    @staticmethod
    def _geography(
        challenge: ChallengeRequirements,
        organization: OrganizationCapabilityProfile,
    ) -> tuple[RawFeature, float | None]:
        if challenge.location is None:
            return RawFeature(0.0, available=False, data_present=False), None
        if organization.location is None:
            return RawFeature(
                0.0,
                gaps=("Organization location was not provided.",),
                available=True,
                data_present=False,
            ), None

        distance = haversine_km(
            challenge.location.latitude,
            challenge.location.longitude,
            organization.location.latitude,
            organization.location.longitude,
        )
        score = max(0.0, 1.0 - distance / 500.0)
        return RawFeature(
            score,
            (f"Organization is {distance:.1f} km from the challenge.",),
            ("Geographic distance may limit field collaboration.",)
            if score < 0.4 else (),
            available=True,
        ), distance
