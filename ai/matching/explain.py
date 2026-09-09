from __future__ import annotations

from ai.schemas.matching import MatchResult


class MatchExplainer:
    """Create deterministic explanations from verified scoring evidence only.

    Explanations are generated only from deterministic scoring results so they
    cannot introduce qualifications or claims that were not produced by the
    scorer.
    """

    async def explain(
        self,
        result: MatchResult,
    ) -> MatchResult:
        explanation = self._truncate(self._deterministic(result), 1000)
        payload = result.model_dump(mode="python", exclude_computed_fields=True)
        payload["explanation"] = explanation
        return MatchResult.model_validate(payload)

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        cleaned = " ".join(value.split()).strip()
        if len(cleaned) <= limit:
            return cleaned
        shortened = cleaned[: limit + 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
        return shortened or cleaned[:limit]

    @staticmethod
    def _deterministic(result: MatchResult) -> str:
        if result.hard_constraint_failures:
            return "Not eligible because " + "; ".join(result.hard_constraint_failures) + "."

        strong = [
            item
            for item in result.feature_breakdown
            if item.available and item.data_present and item.value >= 0.7
        ]
        names = [item.name.replace("_", " ") for item in strong[:3]]

        if result.final_score >= 0.75 and names:
            opening = f"Strong match based on {', '.join(names)}."
        elif result.final_score >= 0.5 and names:
            opening = f"Moderate match based on {', '.join(names)}."
        elif names:
            opening = f"Partial match based on {', '.join(names)}."
        else:
            opening = "Limited capability evidence is available for this match."

        material_gaps = result.gaps[:2]
        if material_gaps:
            return opening + " Main gap: " + "; ".join(material_gaps) + "."
        return opening