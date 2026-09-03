"""Tests for the interview question prompt."""

from talentscout.agents.graph import InterviewerAgent


def test_question_prompt_uses_only_job_and_guidance_context() -> None:
    """Ensure retired company context cannot enter the interviewer prompt."""
    prompt = InterviewerAgent._build_prompt(
        job_description="Build reliable PostgreSQL-backed services.",
        retrieved_context=["Assess connection-pool sizing and failure handling."],
        candidate_answer="I configured a bounded connection pool.",
    )

    assert "Relevant Additional Interview Guidance" in prompt
    assert "connection-pool sizing" in prompt
    assert "Company/project context" not in prompt
    assert "company context" not in prompt.lower()
