"""Tests for validated Role and Competency Agent behaviour."""

from typing import cast

import pytest
from pydantic import BaseModel, ValidationError

from talentscout.agents.role_competency import (
    AssessmentPriority,
    CompetencyRequirement,
    RequirementEvidence,
    RequirementSource,
    RoleCompetencyAgent,
    RoleCompetencyAnalysis,
    RoleCompetencyRequest,
)
from talentscout.llm.service import LLMService, StructuredOutputT


class FakeLLMService(LLMService):
    """Records structured LLM requests without calling an external provider."""

    def __init__(self, response: RoleCompetencyAnalysis) -> None:
        self.response = response
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None
        self.response_model: type[BaseModel] | None = None

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Satisfy the text-generation interface for this test double."""
        raise AssertionError("RoleCompetencyAgent must request structured output")

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredOutputT],
    ) -> StructuredOutputT:
        """Return a predetermined validated analysis."""
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.response_model = response_model
        return cast(StructuredOutputT, self.response)


def build_analysis() -> RoleCompetencyAnalysis:
    """Create a representative validated agent result."""
    return RoleCompetencyAnalysis(
        role_summary="Build and operate reliable Python backend services.",
        required_technologies=["Python", "PostgreSQL"],
        competencies=[
            CompetencyRequirement(
                name="Database scalability",
                description="Design scalable PostgreSQL data access.",
                priority=AssessmentPriority.HIGH,
                source=RequirementSource.BOTH,
                technologies=["PostgreSQL"],
                assessment_focus=["Connection-pool sizing", "Failure handling"],
                evidence=[
                    RequirementEvidence(
                        source=RequirementSource.JOB_DESCRIPTION,
                        excerpt="Build reliable Python services using PostgreSQL.",
                    ),
                    RequirementEvidence(
                        source=RequirementSource.ADDITIONAL_INTERVIEW_GUIDANCE,
                        excerpt="Assess connection-pool sizing and failure handling.",
                    ),
                ],
            )
        ],
        guidance_priorities=["Connection-pool sizing"],
        de_emphasized_areas=[],
    )


@pytest.mark.asyncio
async def test_agent_requests_source_grounded_structured_analysis() -> None:
    """Verify that the agent passes delimited inputs to structured LLM generation."""
    llm = FakeLLMService(build_analysis())
    agent = RoleCompetencyAgent(llm)

    result = await agent.analyze(
        RoleCompetencyRequest(
            job_description="Build reliable Python services using PostgreSQL.",
            additional_guidance=["Assess connection-pool sizing and failure handling."],
        )
    )

    assert result == llm.response
    assert llm.response_model is RoleCompetencyAnalysis
    assert llm.system_prompt is not None
    assert "Do not invent role requirements" in llm.system_prompt
    assert llm.user_prompt is not None
    assert "<job_description>" in llm.user_prompt
    assert "<additional_interview_guidance index=\"1\">" in llm.user_prompt


def test_request_rejects_blank_guidance_chunks() -> None:
    """Avoid wasting an LLM call on invalid optional guidance input."""
    with pytest.raises(ValidationError, match="cannot contain blank chunks"):
        RoleCompetencyRequest(
            job_description="Build reliable Python services.",
            additional_guidance=["   "],
        )


def test_analysis_rejects_unexpected_fields() -> None:
    """Keep the agent contract stable and reject unvalidated LLM data."""
    payload = build_analysis().model_dump()
    payload["unsupported_field"] = "not allowed"

    with pytest.raises(ValidationError, match="unsupported_field"):
        RoleCompetencyAnalysis.model_validate(payload)
