"""Generate the final evidence-based assessment report for an interview."""

import json

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
    evidence: list[str] = Field(max_length=10)
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
        """Generate and validate the final evidence-based assessment."""
        report = await self.llm.generate_structured(
            system_prompt=self._system_prompt(),
            user_prompt=self._build_user_prompt(request),
            response_model=AssessmentReport,
        )

        return self._enforce_evidence_sufficiency(request, report)

    @staticmethod
    def _enforce_evidence_sufficiency(
        request: AssessmentReportRequest,
        report: AssessmentReport,
    ) -> AssessmentReport:
        """Prevent insufficiently assessed competencies from receiving positive ratings."""
        role_competencies = request.role_competency_analysis.competencies
        required_names = {competency.name for competency in role_competencies}

        sufficiently_assessed = {
            name
            for turn in request.interview_history
            for name in turn["evaluation"].sufficiently_assessed_competencies
        }

        assessments_by_name = {
            assessment.name: assessment for assessment in report.competency_assessments
        }

        reported_names = set(assessments_by_name)

        unknown_names = reported_names - required_names
        if unknown_names:
            raise ValueError(
                f"Assessment report contains unknown competencies: {sorted(unknown_names)}"
            )

        duplicate_names = len(report.competency_assessments) != len(reported_names)
        if duplicate_names:
            raise ValueError("Assessment report contains duplicate competency assessments.")

        missing_sufficient_assessments = sufficiently_assessed - reported_names
        if missing_sufficient_assessments:
            raise ValueError(
                "Assessment report omitted competencies with sufficient "
                "interview evidence: "
                f"{sorted(missing_sufficient_assessments)}"
            )

        competency_assessments: list[CompetencyAssessment] = []
        follow_up_areas = list(report.follow_up_areas)

        for competency in role_competencies:
            name = competency.name
            assessment = assessments_by_name.get(name)

            if name in sufficiently_assessed:
                if assessment is None:
                    raise ValueError(
                        f"Missing assessment for sufficiently assessed competency: {name}"
                    )

                competency_assessments.append(assessment)
                continue

            insufficient_gap = (
                "Insufficient interview evidence to make a reliable proficiency assessment."
            )

            if assessment is None:
                assessment = CompetencyAssessment(
                    name=name,
                    performance=EvaluationLevel.INSUFFICIENT,
                    evidence=[],
                    gaps=[insufficient_gap],
                )
            else:
                gaps = list(assessment.gaps)

                if insufficient_gap not in gaps:
                    gaps.append(insufficient_gap)

                assessment = assessment.model_copy(
                    update={
                        "performance": EvaluationLevel.INSUFFICIENT,
                        "gaps": gaps[:10],
                    }
                )

            competency_assessments.append(assessment)

            follow_up_area = f"{name}: additional interview evidence is required."

            if follow_up_area not in follow_up_areas:
                follow_up_areas.append(follow_up_area)

        return report.model_copy(
            update={
                "competency_assessments": competency_assessments,
                "follow_up_areas": follow_up_areas[:10],
            }
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
            "competency assessment must be grounded in the recorded evidence. "
            "IMPORTANT: Distinguish competencies that were merely addressed "
            "from competencies that were sufficiently assessed. Use the "
            "sufficiently_assessed_competencies field in the interview history "
            "as the authoritative record of assessment coverage. If a required "
            "competency was never sufficiently assessed, assign it an "
            "INSUFFICIENT performance rating. Do not assign STRONG or ADEQUATE "
            "ratings without sufficient supporting evidence. Clearly identify "
            "insufficient evidence and relevant follow-up areas. Include every "
            "required role competency in the report, including competencies "
            "that were never addressed. Do not treat the interview reaching "
            "its maximum turn limit as proof that all competencies were "
            "adequately assessed."
        )

    @staticmethod
    def _build_user_prompt(
        request: AssessmentReportRequest,
    ) -> str:
        """Build a compact, grounded report prompt from interview evidence."""
        competency_data = request.role_competency_analysis.model_dump(mode="json")

        history_data = [
            {
                "question": turn["question"],
                "candidate_answer": turn["candidate_answer"],
                "evaluation": turn["evaluation"].model_dump(mode="json"),
            }
            for turn in request.interview_history
        ]

        competency_text = json.dumps(
            competency_data,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        history_text = json.dumps(
            history_data,
            separators=(",", ":"),
            ensure_ascii=False,
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
            "Generate an evidence-based final technical interview assessment. "
            "Include every required competency from the role analysis."
        )
