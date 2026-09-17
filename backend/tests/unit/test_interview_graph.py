"""Unit tests for the LangGraph interview workflow."""

from uuid import uuid4

import pytest

from talentscout.agents.answer_evaluation import (
    AnswerEvaluation,
    AnswerEvaluationAgent,
    EvaluationLevel,
)
from talentscout.agents.graph import build_interview_graph
from talentscout.agents.role_competency import (
    AssessmentPriority,
    CompetencyRequirement,
    RequirementEvidence,
    RequirementSource,
    RoleCompetencyAgent,
    RoleCompetencyAnalysis,
)
from talentscout.agents.state import InterviewState


class FakeRoleCompetencyAgent(RoleCompetencyAgent):
    """Return a deterministic role analysis without calling a real LLM."""

    def __init__(self) -> None:
        """Initialize the fake agent and its call tracking."""
        self.call_count = 0
        self.last_request = None

    async def analyze(self, request) -> RoleCompetencyAnalysis:
        """Return a deterministic competency analysis."""
        self.call_count += 1
        self.last_request = request

        return RoleCompetencyAnalysis(
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


class FakeDocumentChunk:
    """Represent the document-chunk fields needed by the graph test."""

    def __init__(self, text: str) -> None:
        """Initialize a fake retrieved document chunk."""
        self.text = text
        
class FakeRetriever:
    """Return deterministic retrieval results without accessing the database."""

    def __init__(self) -> None:
        """Initialize the fake retriever and call tracking."""
        self.call_count = 0
        self.queries: list[str] = []

    async def retrieve(
        self,
        job_id,
        query: str,
        limit: int = 5,
    ) -> list:
        """Return no guidance while recording the retrieval request."""
        self.call_count += 1
        self.queries.append(query)
        return [
            FakeDocumentChunk(
                text="Focus on designing resilient FastAPI APIs.",
            )
        ]        

class FakeAnswerEvaluationAgent(AnswerEvaluationAgent):
    """Return deterministic answer evaluations without calling an LLM."""

    def __init__(self) -> None:
        """Initialize the fake evaluator and call tracking."""
        self.call_count = 0

    async def evaluate(self, request) -> AnswerEvaluation:
        """Return a deterministic answer evaluation."""
        self.call_count += 1

        return AnswerEvaluation(
            correctness=EvaluationLevel.STRONG,
            depth=EvaluationLevel.ADEQUATE,
            reasoning=EvaluationLevel.STRONG,
            demonstrated_competencies=["Python backend development"],
            strengths=["Clear architecture reasoning."],
            gaps=["Error handling was not discussed."],
            evidence=["Candidate described separating API and service layers."],
            confidence=0.9,
        )
    
class FakeLLM:
    """Return a deterministic interview question without calling an LLM provider."""

    def __init__(self) -> None:
        """Initialize the fake LLM and call tracking."""
        self.call_count = 0
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return a deterministic interview question."""
        self.call_count += 1
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
                
        return "How would you design a FastAPI service for this role?"


@pytest.mark.asyncio
async def test_interview_graph_reuses_initial_retrieval() -> None:
    """Verify that the first question reuses guidance retrieved during role analysis."""
    role_agent = FakeRoleCompetencyAgent()
    retriever = FakeRetriever()
    llm = FakeLLM()

    graph = build_interview_graph(
        retriever=retriever,
        llm=llm,
        role_competency_agent=role_agent,
    )

    job_id = uuid4()

    initial_state: InterviewState = {
        "job_id": job_id,
        "job_description": (
            "Build backend services using Python and FastAPI."
        ),
    }

    result = await graph.ainvoke(initial_state)

    assert role_agent.call_count == 1
    assert retriever.call_count == 1
    assert llm.call_count == 1

    assert role_agent.last_request is not None
    assert role_agent.last_request.additional_guidance == [
        "Focus on designing resilient FastAPI APIs."
    ]

    assert result["retrieved_context"] == [
        "Focus on designing resilient FastAPI APIs."
    ]

    assert result["role_competency_analysis"].role_summary == (
        "Python backend engineer"
    )

    assert result["current_question"] == (
        "How would you design a FastAPI service for this role?"
    )

    assert result["current_question"] == (
        "How would you design a FastAPI service for this role?"
    )

@pytest.mark.asyncio
async def test_interview_graph_handles_initial_and_follow_up_turns() -> None:
    """Verify role analysis runs once and follow-up turns reuse the analysis."""
    role_agent = FakeRoleCompetencyAgent()
    evaluation_agent = FakeAnswerEvaluationAgent()
    retriever = FakeRetriever()
    llm = FakeLLM()

    graph = build_interview_graph(
        retriever=retriever,
        llm=llm,
        role_competency_agent=role_agent,
        answer_evaluation_agent=evaluation_agent,
    )

    job_id = uuid4()

    initial_state: InterviewState = {
        "job_id": job_id,
        "job_description": (
            "Build backend services using Python and FastAPI."
        ),
    }

    first_result = await graph.ainvoke(initial_state)

    # Initial interview: role analysis and retrieval both happen.
    assert role_agent.call_count == 1
    assert retriever.call_count == 1
    assert llm.call_count == 1
    assert evaluation_agent.call_count == 0    

    assert role_agent.last_request is not None
    assert role_agent.last_request.additional_guidance == [
        "Focus on designing resilient FastAPI APIs."
    ]

    assert first_result["retrieved_context"] == [
        "Focus on designing resilient FastAPI APIs."
    ]

    assert first_result["interview_history"] == []

    assert first_result["current_question"] == (
        "How would you design a FastAPI service for this role?"
    )

    assert "No previous answer has been evaluated yet." in (
        llm.user_prompts[0]
    )

    follow_up_state: InterviewState = {
        **first_result,
        "candidate_answer": (
            "I would use a layered architecture with separate API, "
            "service, and repository layers."
        ),
    }

    second_result = await graph.ainvoke(follow_up_state)

    # Follow-up: role analysis must NOT run again.
    assert role_agent.call_count == 1

    # Retrieval and question generation do happen again for adaptation.
    assert retriever.call_count == 2
    assert llm.call_count == 2
    assert evaluation_agent.call_count == 1

    assert second_result["answer_evaluation"].correctness is (
        EvaluationLevel.STRONG
    )  

    assert "Error handling was not discussed." in llm.user_prompts[1]
    assert (
        "Candidate described separating API and service layers."
        in llm.user_prompts[1]
    )

    assert (
        "I would use a layered architecture with separate API, "
        "service, and repository layers."
        in llm.user_prompts[1]
    )          

    assert second_result["interview_history"] == [
        {
            "question": (
                "How would you design a FastAPI service for this role?"
            ),
            "candidate_answer": (
                "I would use a layered architecture with separate API, "
                "service, and repository layers."
            ),
            "evaluation": second_result["answer_evaluation"],
        }
    ]

    assert second_result["current_question"] == (
        "How would you design a FastAPI service for this role?"
    )