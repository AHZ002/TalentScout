"""Generate the final evidence-based assessment report for an interview."""


from pydantic import BaseModel, ConfigDict, Field

from talentscout.agents.answer_evaluation import EvaluationLevel
from talentscout.agents.interview_types import InterviewTurn
from talentscout.agents.role_competency import RoleCompetencyAnalysis
from talentscout.llm.service import LLMService


class CompetencyAssessment(BaseModel):
    """Summarize evidence collected for one assessed competency."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    performance: EvaluationLevel
    evidence: list[str] = Field(min_length=1, max_length=10)
    gaps: list[str] = Field(max_length=10)


class AssessmentReportRequest(BaseModel):
    """Input required to generate the final interview assessment."""

    model_config = ConfigDict(extra="forbid")

    job_description: str = Field(min_length=1, max_length=20_000)
    role_competency_analysis: RoleCompetencyAnalysis
    interview_history: list[InterviewTurn] = Field(min_length=1, max_length=50)


class AssessmentReport(BaseModel):
    """Structured evidence-based result of the technical interview."""

    model_config = ConfigDict(extra="forbid")

    overall_summary: str = Field(min_length=1, max_length=2_000)
    competency_assessments: list[CompetencyAssessment] = Field(
        min_length=1,
        max_length=12,
    )
    strengths: list[str] = Field(max_length=10)
    gaps: list[str] = Field(max_length=10)
    follow_up_areas: list[str] = Field(max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)


class AssessmentReportAgent:
    """Use an LLM to synthesize the evidence collected during an interview."""

    def __init__(self, llm: LLMService) -> None:
        """Initialize the report agent with the configured LLM service."""
        self.llm = llm

    async def generate(
        self,
        request: AssessmentReportRequest,
    ) -> AssessmentReport:
        """Generate a final assessment from the completed interview evidence."""
        return await self.llm.generate_structured(
            system_prompt=self._system_prompt(),
            user_prompt=self._build_user_prompt(request),
            response_model=AssessmentReport,
        )

    @staticmethod
    def _system_prompt() -> str:
        """Provide grounding rules for final assessment generation."""
        return (
            "You are TalentScout's Assessment & Report Agent. Synthesize only "
            "the supplied Job Description, role competency analysis, interview "
            "history, candidate answers, and answer evaluations. Do not invent "
            "candidate experience, skills, evidence, or gaps that are not "
            "supported by the interview record. Distinguish demonstrated "
            "competence from areas where evidence is insufficient. Every "
            "competency assessment must be grounded in the recorded evidence."
        )

    @staticmethod
    def _build_user_prompt(
        request: AssessmentReportRequest,
    ) -> str:
        """Build the grounded report prompt from the completed interview."""
        competency_text = request.role_competency_analysis.model_dump_json(
            indent=2
        )

        history_text = "\n\n".join(
            (
                f"Question: {turn['question']}\n"
                f"Candidate answer: {turn['candidate_answer']}\n"
                "Answer evaluation:\n"
                f"{turn['evaluation'].model_dump_json(indent=2)}"
            )
            for turn in request.interview_history
        )

        return (
            "<job_description>\n"
            f"{request.job_description}\n"
            "</job_description>\n\n"
            "<role_competency_analysis>\n"
            f"{competency_text}\n"
            "</role_competency_analysis>\n\n"
            "<interview_history>\n"
            f"{history_text}\n"
            "</interview_history>\n\n"
            "Generate an evidence-based final technical interview assessment."
        )