from __future__ import annotations

import os
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai.embeddings.gemini import GeminiEmbeddingBackend
from ai.orchestrator import AIOrchestrator

from ai.schemas.dedup import IncomingProblem
from ai.schemas.matching import (
    ChallengeRequirements,
    GeoPoint,
    OrganizationCapabilityProfile,
    OrganizationType,
)

from ai.schemas.workflow import ApprovalDecision


app = FastAPI(
    title="JAN SETU AI Service",
    version="0.1.0",
)


# ============================================================
# EXISTING DATABASE PROBLEM
# ============================================================

class ExistingProblem(BaseModel):
    """
    A previously submitted report supplied by the Go backend.

    Go owns PostgreSQL and sends these existing reports to Python
    only for AI duplicate detection.
    """

    id: str
    title: str
    description: str
    domain: str
    tags: list[str] = Field(default_factory=list)
    locality: str


# ============================================================
# REPORT REQUEST / RESPONSE
# ============================================================

class ReportRequest(BaseModel):
    title: str = Field(
        min_length=3,
        max_length=180,
    )

    description: str = Field(
        min_length=3,
        max_length=10_000,
    )

    locality: str = Field(
        min_length=1,
    )

    district: str = Field(
        min_length=1,
    )

    pin_code: str = Field(
        pattern=r"^\d{6}$",
    )

    # Existing reports fetched by Go from PostgreSQL.
    #
    # Python NEVER accesses PostgreSQL directly.
    existing_problems: list[ExistingProblem] = Field(
        default_factory=list,
    )


class ReportResponse(BaseModel):
    status: str
    decision: str
    decision_reasons: list[str]
    analysis: dict[str, Any]
    problem_statement: dict[str, Any] | None = None
    duplicate_matches: list[dict[str, Any]] = Field(
        default_factory=list,
    )


# ============================================================
# INDUSTRY MATCHING MODELS
# ============================================================

class IndustryOrganization(BaseModel):
    """
    Company / industry capability profile supplied by Go.

    Go owns the database.
    Python only uses this information for matching.
    """

    id: str

    name: str

    organization_type: OrganizationType = (
        OrganizationType.INDUSTRY
    )

    description: str = ""

    domains: list[str] = Field(
        default_factory=list,
    )

    skills: list[str] = Field(
        default_factory=list,
    )

    research_areas: list[str] = Field(
        default_factory=list,
    )

    facilities: list[str] = Field(
        default_factory=list,
    )

    industry_capabilities: list[str] = Field(
        default_factory=list,
    )

    verified: bool = False

    location: GeoPoint | None = None


class IndustryProblem(BaseModel):
    """
    Civic problem supplied by Go for industry matching.

    Go owns the database.
    Python converts this into the existing
    ChallengeRequirements model.
    """

    id: str

    title: str = Field(
        min_length=3,
        max_length=180,
    )

    description: str = Field(
        min_length=3,
        max_length=10_000,
    )

    category: str = ""

    domain: str | None = None

    locality: str = ""

    district: str = ""

    pin_code: str = ""

    # These allow the Go backend to pass AI-derived
    # requirements later without changing this endpoint.
    skills: list[str] = Field(
        default_factory=list,
    )

    required_capabilities: list[str] = Field(
        default_factory=list,
    )

    required_facilities: list[str] = Field(
        default_factory=list,
    )

    preferred_industry_capabilities: list[str] = Field(
        default_factory=list,
    )

    constraints: list[str] = Field(
        default_factory=list,
    )


class IndustryMatchRequest(BaseModel):
    organization: IndustryOrganization

    problems: list[IndustryProblem] = Field(
        default_factory=list,
    )


class IndustryMatchResponse(BaseModel):
    matches: list[dict[str, Any]] = Field(
        default_factory=list,
    )


# ============================================================
# GLOBAL ORCHESTRATOR
# ============================================================

_orchestrator: AIOrchestrator | None = None


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup() -> None:
    global _orchestrator

    embedding_backend = GeminiEmbeddingBackend(
        api_key=os.getenv(
            "JANSETU_EMBEDDING_API_KEY"
        ),
    )

    _orchestrator = await AIOrchestrator.build(
        embedding_backend=embedding_backend,
        problems=[],
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


# ============================================================
# REPORT ANALYSIS
# ============================================================

@app.post(
    "/analyze-report",
    response_model=ReportResponse,
)
async def analyze_report(
    request: ReportRequest,
) -> ReportResponse:

    if _orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="AI service is not initialized",
        )

    report = (
        f"{request.title}\n\n"
        f"{request.description}"
    )

    context = {
        "locality": request.locality,
        "district": request.district,
        "pin_code": request.pin_code,
    }

    try:

        # ========================================================
        # 1. AI UNDERSTANDING + VALIDATION
        # ========================================================

        workflow = await _orchestrator.process_citizen_report(
            report=report,
            context=context,
        )

        # ========================================================
        # 2. HANDLE REPORTS THAT FAILED VALIDATION
        # ========================================================

        if (
            workflow.decision
            is not ApprovalDecision.AUTO_APPROVED
        ):
            return ReportResponse(
                status=_map_decision_to_status(
                    workflow.decision
                ),
                decision=workflow.decision.value,
                decision_reasons=(
                    workflow.decision_reasons
                ),
                analysis=(
                    workflow.analysis.model_dump(
                        mode="json"
                    )
                ),
                problem_statement=None,
            )

        # ========================================================
        # 3. CONVERT APPROVED REPORT INTO DEDUP INPUT
        # ========================================================

        statement = workflow.problem_statement

        if statement is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Approved report has no "
                    "problem statement"
                ),
            )

        problem = IncomingProblem(
            title=statement.title,
            description=statement.problem_summary,
            domain=workflow.analysis.domain,
            tags=workflow.analysis.tags,
            locality=request.locality,
        )

        # ========================================================
        # 4. BUILD DEDUP ORCHESTRATOR WITH EXISTING REPORTS
        # ========================================================

        existing_problems = [
            IncomingProblem(
                id=existing.id,
                title=existing.title,
                description=existing.description,
                domain=existing.domain,
                tags=existing.tags,
                locality=existing.locality,
            )
            for existing in request.existing_problems
        ]

        if existing_problems:

            embedding_backend = (
                GeminiEmbeddingBackend(
                    api_key=os.getenv(
                        "JANSETU_EMBEDDING_API_KEY"
                    ),
                )
            )

            dedup_orchestrator = (
                await AIOrchestrator.build(
                    embedding_backend=embedding_backend,
                    problems=existing_problems,
                )
            )

        else:
            dedup_orchestrator = _orchestrator

        # ========================================================
        # 5. DUPLICATE CHECK
        # ========================================================

        dedup_result = (
            await dedup_orchestrator.find_duplicates(
                problem
            )
        )

        # ========================================================
        # 6. DETERMINE FINAL PRODUCT STATUS
        # ========================================================

        if dedup_result.duplicate_candidates:

            status = "DUPLICATE_EXISTS"

        elif dedup_result.needs_human_review:

            status = "NEED_HUMAN_REVIEW"

        else:

            status = "VALID"

        # ========================================================
        # 7. RETURN STRUCTURED RESPONSE
        # ========================================================

        return ReportResponse(
            status=status,
            decision=workflow.decision.value,
            decision_reasons=(
                workflow.decision_reasons
            ),
            analysis=(
                workflow.analysis.model_dump(
                    mode="json"
                )
            ),
            problem_statement=(
                statement.model_dump(
                    mode="json"
                )
                if status == "VALID"
                else None
            ),
            duplicate_matches=[
                match.model_dump(
                    mode="json"
                )
                for match in dedup_result.matches
            ],
        )

    except HTTPException:
        raise

    except Exception as exc:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "AI processing failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc


# ============================================================
# INDUSTRY AI MATCHING
# ============================================================

@app.post(
    "/match-industry",
    response_model=IndustryMatchResponse,
)
async def match_industry(
    request: IndustryMatchRequest,
) -> IndustryMatchResponse:

    if _orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="AI service is not initialized",
        )

    if not request.problems:
        return IndustryMatchResponse(
            matches=[]
        )

    try:

        # ========================================================
        # 1. CONVERT COMPANY PROFILE
        # ========================================================

        organization = (
            OrganizationCapabilityProfile(
                id=request.organization.id,
                name=request.organization.name,
                organization_type=(
                    request.organization
                    .organization_type
                ),
                description=(
                    request.organization
                    .description
                ),
                domains=(
                    request.organization.domains
                ),
                skills=(
                    request.organization.skills
                ),
                research_areas=(
                    request.organization
                    .research_areas
                ),
                facilities=(
                    request.organization
                    .facilities
                ),
                industry_capabilities=(
                    request.organization
                    .industry_capabilities
                ),
                verified=(
                    request.organization.verified
                ),
                location=(
                    request.organization.location
                ),
            )
        )

        # ========================================================
        # 2. BUILD MATCHING ORCHESTRATOR
        # ========================================================
        #
        # We reuse the EXISTING MatchingService through
        # AIOrchestrator.
        #
        # The company supplied by Go becomes the organization
        # candidate inside the existing organization retriever.
        #
        # ========================================================

        embedding_backend = (
            GeminiEmbeddingBackend(
                api_key=os.getenv(
                    "JANSETU_EMBEDDING_API_KEY"
                ),
            )
        )

        matching_orchestrator = (
            await AIOrchestrator.build(
                embedding_backend=embedding_backend,
                organizations=[organization],
            )
        )

        # ========================================================
        # 3. MATCH EVERY CIVIC PROBLEM
        # ========================================================

        matches: list[dict[str, Any]] = []

        for problem in request.problems:

            # ----------------------------------------------------
            # Build a useful summary.
            #
            # The existing ChallengeRequirements model does not
            # have locality/district/pin_code string fields.
            #
            # Including the location in the summary preserves
            # that context for semantic matching.
            # ----------------------------------------------------

            location_parts = [
                value.strip()
                for value in [
                    problem.locality,
                    problem.district,
                    problem.pin_code,
                ]
                if value
            ]

            if location_parts:

                location_context = (
                    "Location: "
                    + ", ".join(
                        location_parts
                    )
                )

                summary = (
                    f"{problem.description}\n"
                    f"{location_context}"
                )

            else:

                summary = problem.description

            # ChallengeRequirements has a maximum summary
            # length of 1000 characters.
            summary = summary[:1000]

            # ----------------------------------------------------
            # Build existing matching schema.
            # ----------------------------------------------------

            challenge = ChallengeRequirements(
                id=problem.id,

                title=problem.title,

                summary=summary,

                domain=(
                    problem.domain
                    or problem.category
                    or None
                ),

                skills=problem.skills,

                required_capabilities=(
                    problem.required_capabilities
                ),

                required_facilities=(
                    problem.required_facilities
                ),

                preferred_industry_capabilities=(
                    problem
                    .preferred_industry_capabilities
                ),

                constraints=problem.constraints,
            )

            # ----------------------------------------------------
            # USE EXISTING AI MATCHING PIPELINE
            # ----------------------------------------------------

            result = (
                await matching_orchestrator
                .match_organizations(
                    challenge
                )
            )

            # MatchingResultSet contains a list of MatchResult
            # objects in result.matches.
            for match in result.matches:

                match_data = match.model_dump(
                    mode="json"
                )

                # Add the database/report information that
                # the frontend needs.
                match_data["problem_id"] = (
                    problem.id
                )

                match_data["problem_title"] = (
                    problem.title
                )

                match_data["category"] = (
                    problem.category
                )

                match_data["locality"] = (
                    problem.locality
                )

                match_data["district"] = (
                    problem.district
                )

                matches.append(
                    match_data
                )

        # ========================================================
        # 4. SORT BY EXISTING FINAL AI SCORE
        # ========================================================

        matches.sort(
            key=lambda item: item.get(
                "final_score",
                0,
            ),
            reverse=True,
        )

        return IndustryMatchResponse(
            matches=matches
        )

    except HTTPException:
        raise

    except Exception as exc:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Industry matching failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc


# ============================================================
# DECISION → PRODUCT STATUS
# ============================================================

def _map_decision_to_status(
    decision: ApprovalDecision,
) -> str:

    if (
        decision
        is ApprovalDecision.AUTO_REJECTED
    ):
        return "INVALID"

    if (
        decision
        is ApprovalDecision.CITIZEN_REVISION_REQUIRED
    ):
        return "NEED_HUMAN_REVIEW"

    if (
        decision
        is ApprovalDecision.HUMAN_REVIEW
    ):
        return "NEED_HUMAN_REVIEW"

    return "VALID"