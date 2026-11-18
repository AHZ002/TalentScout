"""Tests for the final interview assessment report agent."""

import pytest

from talentscout.agents.answer_evaluation import (
    AnswerEvaluation,
    EvaluationLevel,
    InterviewDecision,
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
            overall_summary=("The candidate demonstrated strong backend fundamentals."),
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


def build_competency(
    name: str,
    description: str,
) -> CompetencyRequirement:
    """Build a deterministic role competency for report tests."""
    return CompetencyRequirement(
        name=name,
        description=description,
        priority=AssessmentPriority.HIGH,
        source=RequirementSource.JOB_DESCRIPTION,
        technologies=["Python"],
        assessment_focus=["Technical implementation"],
        evidence=[
            RequirementEvidence(
                source=RequirementSource.JOB_DESCRIPTION,
                excerpt=description,
            )
        ],
    )


def build_role_analysis(
    include_database_competency: bool = False,
) -> RoleCompetencyAnalysis:
    """Build the role competency analysis used by report tests."""
    competencies = [
        build_competency(
            name="Python backend development",
            description="Ability to build backend services using Python.",
        )
    ]

    if include_database_competency:
        competencies.append(
            build_competency(
                name="Database design",
                description="Ability to design reliable database schemas.",
            )
        )

    return RoleCompetencyAnalysis(
        role_summary="Python backend engineer",
        required_technologies=["Python", "FastAPI"],
        competencies=competencies,
        guidance_priorities=[],
        de_emphasized_areas=[],
    )


def build_evaluation(
    sufficiently_assessed_competencies: list[str],
    assessed_competencies: list[str] | None = None,
) -> AnswerEvaluation:
    """Build deterministic answer evaluation evidence."""
    return AnswerEvaluation(
        correctness=EvaluationLevel.STRONG,
        depth=EvaluationLevel.ADEQUATE,
        reasoning=EvaluationLevel.STRONG,
        demonstrated_competencies=["Python backend development"],
        strengths=["Clear architecture reasoning."],
        gaps=["Error handling was not discussed."],
        evidence=["Candidate described layered API design."],
        assessed_competencies=(
            assessed_competencies
            if assessed_competencies is not None
            else ["Python backend development"]
        ),
        sufficiently_assessed_competencies=sufficiently_assessed_competencies,
        next_action=InterviewDecision.CONTINUE,
        decision_reason="Further evidence is useful.",
        confidence=0.9,
    )


def build_request(
    role_analysis: RoleCompetencyAnalysis,
    evaluation: AnswerEvaluation,
) -> AssessmentReportRequest:
    """Build a deterministic report request."""
    interview_history: list[InterviewTurn] = [
        {
            "question": "How would you design a FastAPI service?",
            "candidate_answer": ("I would separate the API, service, and repository layers."),
            "evaluation": evaluation,
        }
    ]

    return AssessmentReportRequest(
        job_description="Build backend services using Python and FastAPI.",
        role_competency_analysis=role_analysis,
        interview_history=interview_history,
    )


@pytest.mark.asyncio
async def test_assessment_report_agent_returns_structured_report() -> None:
    """Verify that the report agent produces validated structured output."""
    llm = FakeLLM()
    agent = AssessmentReportAgent(llm=llm)

    role_analysis = build_role_analysis()
    evaluation = build_evaluation(sufficiently_assessed_competencies=["Python backend development"])

    result = await agent.generate(build_request(role_analysis, evaluation))

    assert llm.call_count == 1
    assert llm.response_model is AssessmentReport
    assert result.overall_summary.startswith("The candidate demonstrated")
    assert result.competency_assessments[0].name == ("Python backend development")
    assert result.competency_assessments[0].performance is (EvaluationLevel.STRONG)


@pytest.mark.asyncio
async def test_report_marks_addressed_but_insufficient_competency() -> None:
    """Ensure addressing a competency does not imply sufficient evidence."""
    llm = FakeLLM()
    agent = AssessmentReportAgent(llm=llm)

    role_analysis = build_role_analysis()
    evaluation = build_evaluation(
        assessed_competencies=["Python backend development"],
        sufficiently_assessed_competencies=[],
    )

    result = await agent.generate(build_request(role_analysis, evaluation))

    assessment = result.competency_assessments[0]

    assert assessment.name == "Python backend development"
    assert assessment.performance is EvaluationLevel.INSUFFICIENT
    assert any("Insufficient interview evidence" in gap for gap in assessment.gaps)
    assert any("Python backend development" in area for area in result.follow_up_areas)


@pytest.mark.asyncio
async def test_report_includes_competencies_never_addressed() -> None:
    """Ensure every required competency appears in the final report."""
    llm = FakeLLM()
    agent = AssessmentReportAgent(llm=llm)

    role_analysis = build_role_analysis(include_database_competency=True)
    evaluation = build_evaluation(
        assessed_competencies=["Python backend development"],
        sufficiently_assessed_competencies=["Python backend development"],
    )

    result = await agent.generate(build_request(role_analysis, evaluation))

    assessments = {assessment.name: assessment for assessment in result.competency_assessments}

    assert set(assessments) == {
        "Python backend development",
        "Database design",
    }

    assert assessments["Python backend development"].performance is (EvaluationLevel.STRONG)

    assert assessments["Database design"].performance is (EvaluationLevel.INSUFFICIENT)

    assert assessments["Database design"].evidence == []

    assert any("Database design" in area for area in result.follow_up_areas)
