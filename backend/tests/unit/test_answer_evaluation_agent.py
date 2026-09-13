"""Tests for structured candidate-answer evaluation."""

import pytest

from talentscout.agents.answer_evaluation import (
    AnswerEvaluation,
    AnswerEvaluationAgent,
    AnswerEvaluationRequest,
    EvaluationLevel,
)
from talentscout.agents.role_competency import (
    AssessmentPriority,
    CompetencyRequirement,
    RequirementEvidence,
    RequirementSource,
    RoleCompetencyAnalysis,
)


class FakeLLM:
    """Return deterministic structured evaluation output."""

    def __init__(self) -> None:
        """Initialize call tracking."""
        self.call_count = 0
        self.response_model = None

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[AnswerEvaluation],
    ) -> AnswerEvaluation:
        """Return a deterministic structured evaluation."""
        self.call_count += 1
        self.response_model = response_model

        return AnswerEvaluation(
            correctness=EvaluationLevel.STRONG,
            depth=EvaluationLevel.ADEQUATE,
            reasoning=EvaluationLevel.STRONG,
            demonstrated_competencies=["Python backend development"],
            strengths=["Clear layered architecture reasoning."],
            gaps=["Did not discuss error handling."],
            evidence=[
                "Candidate described separating API, service, and repository layers."
            ],
            confidence=0.9,
        )


@pytest.mark.asyncio
async def test_answer_evaluation_agent_returns_structured_evaluation() -> None:
    """Verify that the evaluation agent returns validated structured output."""
    llm = FakeLLM()
    agent = AnswerEvaluationAgent(llm=llm)

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

    result = await agent.evaluate(
        AnswerEvaluationRequest(
            question="How would you design a FastAPI service?",
            candidate_answer=(
                "I would separate the API, service, and repository layers."
            ),
            role_competency_analysis=role_analysis,
        )
    )

    assert llm.call_count == 1
    assert llm.response_model is AnswerEvaluation
    assert result.correctness is EvaluationLevel.STRONG
    assert result.demonstrated_competencies == [
        "Python backend development"
    ]
    assert result.confidence == 0.9