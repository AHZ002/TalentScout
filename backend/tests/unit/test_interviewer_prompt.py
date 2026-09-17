"""Tests for the interview question prompt."""

from talentscout.agents.graph import InterviewerAgent
from talentscout.agents.role_competency import (
    AssessmentPriority,
    CompetencyRequirement,
    RequirementEvidence,
    RequirementSource,
    RoleCompetencyAnalysis,
)


def test_question_prompt_uses_job_guidance_and_role_competencies() -> None:
    """Ensure the interviewer prompt contains all relevant interview context."""
    role_competency_analysis = RoleCompetencyAnalysis(
        role_summary="Build reliable PostgreSQL-backed services.",
        required_technologies=["PostgreSQL"],
        competencies=[
            CompetencyRequirement(
                name="Database scalability",
                description="Design and operate scalable PostgreSQL data access.",
                priority=AssessmentPriority.HIGH,
                source=RequirementSource.BOTH,
                technologies=["PostgreSQL"],
                assessment_focus=[
                    "Connection pooling",
                    "Failure handling",
                ],
                evidence=[
                    RequirementEvidence(
                        source=RequirementSource.JOB_DESCRIPTION,
                        excerpt="Build reliable PostgreSQL-backed services.",
                    ),
                    RequirementEvidence(
                        source=RequirementSource.ADDITIONAL_INTERVIEW_GUIDANCE,
                        excerpt="Assess connection-pool sizing and failure handling.",
                    ),
                ],
            )
        ],
        guidance_priorities=["PostgreSQL connection pooling and failure handling"],
        de_emphasized_areas=[],
    )

    prompt = InterviewerAgent._build_prompt(
        job_description=(
            "Build reliable PostgreSQL-backed services."
        ),
        role_competency_analysis=role_competency_analysis,
        retrieved_context=[
            "Assess connection-pool sizing and failure handling.",
        ],
        candidate_answer="No previous answer yet.",
        interview_history=[],
        answer_evaluation=None,
    )

    assert "Required role competencies" in prompt
    assert "Database scalability" in prompt
    assert "PostgreSQL" in prompt
    assert "Connection pooling" in prompt
    assert "Relevant Additional Interview Guidance" in prompt
    assert "connection-pool sizing" in prompt
    assert "Company/project context" not in prompt
    assert "company context" not in prompt.lower()