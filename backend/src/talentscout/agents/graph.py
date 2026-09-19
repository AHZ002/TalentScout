"""LangGraph workflow for generating context-aware technical interview questions."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from talentscout.agents.answer_evaluation import (
    AnswerEvaluation,
    AnswerEvaluationAgent,
    AnswerEvaluationRequest,
)
from talentscout.agents.assessment_report import (
    AssessmentReportAgent,
    AssessmentReportRequest,
)
from talentscout.agents.role_competency import (
    RoleCompetencyAgent,
    RoleCompetencyAnalysis,
    RoleCompetencyRequest,
)
from talentscout.agents.state import InterviewState
from talentscout.documents.retriever import DocumentRetriever
from talentscout.llm.service import LLMService


async def analyze_role(
    state: InterviewState,
    role_competency_agent: RoleCompetencyAgent,
) -> InterviewState:
    """Analyse the job using the JD and retrieved interview guidance."""
    analysis = await role_competency_agent.analyze(
        RoleCompetencyRequest(
            job_description=state["job_description"],
            additional_guidance=state.get("additional_guidance", []),
        )
    )

    return {
        **state,
        "role_competency_analysis": analysis,
    }

async def evaluate_answer(
    state: InterviewState,
    answer_evaluation_agent: AnswerEvaluationAgent,
) -> InterviewState:
    """Evaluate and record the candidate's latest completed interview turn."""
    evaluation = await answer_evaluation_agent.evaluate(
        AnswerEvaluationRequest(
            question=state["current_question"],
            candidate_answer=state["candidate_answer"],
            role_competency_analysis=state["role_competency_analysis"],
        )
    )

    # The answer is now a completed interview turn because it has been evaluated.
    interview_history = list(state.get("interview_history", []))
    interview_history.append(
        {
            "question": state["current_question"],
            "candidate_answer": state["candidate_answer"],
            "evaluation": evaluation,
        }
    )

    return {
        **state,
        "answer_evaluation": evaluation,
        "interview_history": interview_history,
    }

class InterviewerAgent:
    """Generate an interview question using job and document context."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm: LLMService,
    ) -> None:
        """Initialize the interviewer with retrieval and LLM services."""
        # Retrieval provides relevant Additional Interview Guidance.
        self.retriever = retriever

        # The LLM service handles communication with the configured model.
        self.llm = llm

    async def generate_question(
        self,
        state: InterviewState,
    ) -> InterviewState:
        """Retrieve relevant context and generate the next interview question."""
        job_id = state["job_id"]
        job_description = state["job_description"]
        candidate_answer = state.get("candidate_answer", "")
        role_competency_analysis = state["role_competency_analysis"]
        interview_history = state.get("interview_history", [])
        answer_evaluation = state.get("answer_evaluation")

        # For the first question, retrieve documents using the JD.
        # For later questions, use the JD together with the candidate's
        # previous answer so retrieval remains relevant to the job and
        # adapts to what the candidate has already discussed.
        if candidate_answer.strip():
            # Later questions retrieve fresh guidance using the candidate's
            # previous answer so the interview can adapt to the conversation.
            retrieval_query = (
                f"Job requirements:\n{job_description}\n\n"
                f"Candidate's previous answer:\n{candidate_answer}"
            )

            chunks = await self.retriever.retrieve(
                job_id=job_id,
                query=retrieval_query,
                limit=5,
            )

            retrieved_context = [chunk.text for chunk in chunks]
        else:
            # The role-analysis node already performed the initial retrieval.
            # Reuse it rather than generating another Jina embedding.
            retrieved_context = state.get("retrieved_context", [])

        prompt = self._build_prompt(
            job_description=job_description,
            role_competency_analysis=role_competency_analysis,
            retrieved_context=retrieved_context,
            candidate_answer=candidate_answer,
            interview_history=interview_history,
            answer_evaluation=answer_evaluation,
        )

        # Generate the question through the provider-independent LLM service.
        question = await self.llm.generate(
            system_prompt=(
                "You are TalentScout's technical interviewer. "
                "Generate one clear technical interview question. "
                "Base the question on the job description, role competencies, "
                "relevant retrieved Additional Interview Guidance, interview "
                "history, and the latest answer evaluation. "
                "Adapt the next question to the candidate's demonstrated "
                "strengths and remaining gaps. Prefer probing an unresolved "
                "competency or gap over repeating a previously covered area. "
                "Use the candidate's previous answer when available to make "
                "the next question relevant and appropriately challenging. "
                "Do not invent company-specific facts."
            ),
            user_prompt=prompt,
        )

        return {
            **state,
            "retrieved_context": retrieved_context,
            "current_question": question,
            "interview_history": interview_history,
        }

    @staticmethod
    def _build_prompt(
        job_description: str,
        role_competency_analysis: RoleCompetencyAnalysis,
        retrieved_context: list[str],
        candidate_answer: str,
        interview_history: list,
        answer_evaluation: AnswerEvaluation | None,
    ) -> str:
        """Build the interviewer prompt from the complete interview context."""
        # Combine retrieved guidance chunks into a single context section.
        document_text = "\n\n".join(retrieved_context)

        # Serialize the structured role analysis so the interviewer can
        # select questions against explicit competencies and priorities.
        competency_text = role_competency_analysis.model_dump_json(indent=2)

        # Serialize completed turns so the interviewer knows what has
        # already been discussed and avoids asking redundant questions.
        history_text = (
            "\n\n".join(
                (
                    f"Question: {turn['question']}\n"
                    f"Candidate answer: {turn['candidate_answer']}\n"
                    f"Evaluation: "
                    f"{turn['evaluation'].model_dump_json(indent=2)}"
                )
                for turn in interview_history
            )
            or "No completed interview turns yet."
        )

        # Include the latest evaluation separately because it is the most
        # important signal for deciding what the next question should probe.
        evaluation_text = (
            answer_evaluation.model_dump_json(indent=2)
            if answer_evaluation is not None
            else "No previous answer has been evaluated yet."
        )

        return (
            f"Job description:\n{job_description}\n\n"
            f"Required role competencies:\n{competency_text}\n\n"
            f"Relevant Additional Interview Guidance:\n"
            f"{document_text or 'No relevant guidance was retrieved.'}\n\n"
            f"Interview history:\n{history_text}\n\n"
            f"Latest answer evaluation:\n{evaluation_text}\n\n"
            f"Candidate's latest answer:\n"
            f"{candidate_answer or 'No previous answer; this is the first question.'}\n\n"
            "Generate the next technical interview question. "
            "Use the evaluation and remaining gaps to probe areas that "
            "still need evidence. Do not repeat questions already covered."
        )



def build_interview_graph(
    retriever: DocumentRetriever,
    llm: LLMService,
    role_competency_agent: RoleCompetencyAgent | None = None,
    answer_evaluation_agent: AnswerEvaluationAgent | None = None,
    assessment_report_agent: AssessmentReportAgent | None = None,
) -> Any:
    """Build the initial LangGraph interview workflow."""
    # Allow tests and future callers to inject a controlled Role Competency Agent.
    # Production callers can omit it and use the normal LLM-backed implementation.
    interviewer = InterviewerAgent(
        retriever=retriever,
        llm=llm,
    )

    role_agent = role_competency_agent or RoleCompetencyAgent(llm=llm)

    evaluation_agent = (
        answer_evaluation_agent or AnswerEvaluationAgent(llm=llm)
    )

    report_agent = (
        assessment_report_agent or AssessmentReportAgent(llm=llm)
    )    

    graph = StateGraph(InterviewState)

    async def role_competency_node(
        state: InterviewState,
    ) -> InterviewState:
        """Retrieve guidance and analyse the role before interviewing."""
        # Use the Job Description as the initial semantic retrieval query.
        # If no Additional Interview Guidance exists, the retriever returns
        # an empty list without calling the embedding provider.
        chunks = await retriever.retrieve(
            job_id=state["job_id"],
            query=state["job_description"],
            limit=5,
        )

        retrieved_context = [chunk.text for chunk in chunks]

        # The Role & Competency Agent receives the retrieved guidance as
        # assessment context rather than receiving raw database objects.
        role_state: InterviewState = {
            **state,
            "retrieved_context": retrieved_context,
            "additional_guidance": retrieved_context,
        }

        return await analyze_role(
            role_state,
            role_agent,
        )

    async def answer_evaluation_node(
        state: InterviewState,
    ) -> InterviewState:
        """Evaluate the latest answer before generating the next question."""
        return await evaluate_answer(
            state,
            evaluation_agent,
        )

    async def assessment_report_node(
        state: InterviewState,
    ) -> InterviewState:
        """Generate the final report from the completed interview evidence."""
        report = await report_agent.generate(
            AssessmentReportRequest(
                job_description=state["job_description"],
                role_competency_analysis=state["role_competency_analysis"],
                interview_history=state["interview_history"],
            )
        )

        return {
            **state,
            "assessment_report": report,
        }    

    graph.add_node(
        "role_competency",
        role_competency_node,
    )

    graph.add_node(
        "answer_evaluation",
        answer_evaluation_node,
    )   

    graph.add_node(
        "assessment_report",
        assessment_report_node,
    )     

    # Generate the interview question using the role analysis and retrieved context.
    graph.add_node(
        "interviewer",
        interviewer.generate_question,
    )

    def route_interview(state: InterviewState) -> str:
        """Route initial and follow-up interview turns to the correct nodes."""
        if "role_competency_analysis" not in state:
            return "initial"

        if state.get("candidate_answer"):
            return "follow_up"

        return "resume"

    def route_after_evaluation(state: InterviewState) -> str:
        """Continue interviewing or generate the final report."""
        if state.get("interview_complete", False):
            return "report"

        return "interviewer"        

    graph.add_conditional_edges(
        START,
        route_interview,
        {
            "initial": "role_competency",
            "follow_up": "answer_evaluation",
            "resume": "interviewer",
        },
    )

    graph.add_edge("role_competency", "interviewer")

    graph.add_conditional_edges(
        "answer_evaluation",
        route_after_evaluation,
        {
            "report": "assessment_report",
            "interviewer": "interviewer",
        },
    )

    graph.add_edge("interviewer", END)
    graph.add_edge("assessment_report", END)

    return graph.compile()
