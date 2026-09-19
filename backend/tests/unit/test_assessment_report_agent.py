"""Tests for the final interview assessment report agent."""

import pytest

from talentscout.agents.answer_evaluation import (
    AnswerEvaluation,
    EvaluationLevel,
)
from talentscout.agents.assessment_report import (
    AssessmentReport,
    AssessmentReportAgent,
    AssessmentReportRequest,
    CompetencyAssessment,
)
from talentscout.agents.role_competency import (
    AssessmentPriority,
    CompetencyRequirement,
    RequirementEvidence,
    RequirementSource,
    RoleCompetencyAnalysis,
)
from talentscout.agents.state import InterviewTurn


class FakeLLM:
    """Return deterministic structured assessment output."""

    def __init__(self) -> None:
        """Initialize call tracking."""
        self.call_count = 0
        self.response_model = None

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[AssessmentReport],
    ) -> AssessmentReport:
        """Return a deterministic final assessment."""
        self.call_count += 1
        self.response_model = response_model

        return AssessmentReport(
            overall_summary="The candidate demonstrated strong backend fundamentals.",
            competency_assessments=[
                CompetencyAssessment(
                    name="Python backend development",
                    performance=EvaluationLevel.STRONG,
                    evidence=["Candidate described layered API design."],
                    gaps=["Error handling was not discussed."],
                )
            ],
            strengths=["Clear backend architecture reasoning."],
            gaps=["Error handling depth remains insufficient."],
            follow_up_areas=["Exception handling and resilience."],
            confidence=0.9,
        )


@pytest.mark.asyncio
async def test_assessment_report_agent_returns_structured_report() -> None:
    """Verify that the report agent produces validated structured output."""
    llm = FakeLLM()
    agent = AssessmentReportAgent(llm=llm)

    role_analysis = RoleCompetencyAnalysis(
        role_summary="Python backend engineer",
        required_technologies=["Python", "FastAPI"],
        competencies=[
            CompetencyRequirement(
                name="Python backend development",
                description="Ability to build backend services with Python.",
                priority=AssessmentPriority.HIGH,
                source=RequirementSource.JOB_DESCRIPTION,
                technologies=["Python"],
                assessment_focus=["API development"],
                evidence=[
                    RequirementEvidence(
                        source=RequirementSource.JOB_DESCRIPTION,
                        excerpt="Build backend services using Python.",
                    )
                ],
            )
        ],
        guidance_priorities=[],
        de_emphasized_areas=[],
    )

    evaluation = AnswerEvaluation(
        correctness=EvaluationLevel.STRONG,
        depth=EvaluationLevel.ADEQUATE,
        reasoning=EvaluationLevel.STRONG,
        demonstrated_competencies=["Python backend development"],
        strengths=["Clear architecture reasoning."],
        gaps=["Error handling was not discussed."],
        evidence=["Candidate described layered API design."],
        confidence=0.9,
    )

    interview_history: list[InterviewTurn] = [
        {
            "question": "How would you design a FastAPI service?",
            "candidate_answer": (
                "I would separate the API, service, and repository layers."
            ),
            "evaluation": evaluation,
        }
    ]

    result = await agent.generate(
        AssessmentReportRequest(
            job_description="Build backend services using Python and FastAPI.",
            role_competency_analysis=role_analysis,
            interview_history=interview_history,
        )
    )

    assert llm.call_count == 1
    assert llm.response_model is AssessmentReport
    assert result.overall_summary.startswith("The candidate demonstrated")
    assert result.competency_assessments[0].name == (
        "Python backend development"
    )