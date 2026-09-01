"""LangGraph workflow for generating context-aware technical interview questions."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from talentscout.agents.state import InterviewState
from talentscout.documents.retriever import DocumentRetriever
from talentscout.llm.service import LLMService


class InterviewerAgent:
    """Generate an interview question using job-specific retrieved context."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm: LLMService,
    ) -> None:
        """Initialize the interviewer with retrieval and LLM services."""
        # Retrieval provides company and job-specific context.
        self.retriever = retriever

        # The LLM service handles communication with the configured model.
        self.llm = llm

    async def generate_question(
        self,
        state: InterviewState,
    ) -> InterviewState:
        """Retrieve relevant context and generate the next interview question."""
        job_id = state["job_id"]

        # Use the candidate's previous answer for contextual retrieval.
        # For the first question, use a general technical-screening query.
        query = state.get(
            "candidate_answer",
            "technical requirements, systems, technologies, and domain knowledge",
        )

        chunks = await self.retriever.retrieve(
            job_id=job_id,
            query=query,
            limit=5,
        )

        context = [chunk.text for chunk in chunks]

        prompt = self._build_prompt(
            context=context,
            candidate_answer=state.get("candidate_answer", ""),
        )

        # Generate the question through the provider-independent LLM service.
        question = await self.llm.generate(
            system_prompt=(
                "You are TalentScout's technical interviewer. "
                "Generate one clear technical interview question. "
                "Use the supplied company and job context. "
                "Do not invent company-specific facts."
            ),
            user_prompt=prompt,
        )

        return {
            **state,
            "retrieved_context": context,
            "current_question": question,
        }

    @staticmethod
    def _build_prompt(
        context: list[str],
        candidate_answer: str,
    ) -> str:
        """Build the interviewer prompt from context and candidate state."""
        # Combine retrieved chunks into a single context section for the LLM.
        context_text = "\n\n".join(context)

        return (
            f"Relevant company/job context:\n{context_text}\n\n"
            f"Candidate's previous answer:\n"
            f"{candidate_answer or 'No previous answer.'}\n\n"
            "Generate the next technical interview question."
        )


def build_interview_graph(
    retriever: DocumentRetriever,
    llm: LLMService,
) -> Any:
    """Build the initial LangGraph interview workflow."""
    # The interviewer agent coordinates retrieval and question generation.
    interviewer = InterviewerAgent(
        retriever=retriever,
        llm=llm,
    )

    # Define the shared interview state used by the graph.
    graph = StateGraph(InterviewState)

    # The first graph node retrieves context and generates a question.
    graph.add_node(
        "interviewer",
        interviewer.generate_question,
    )

    # Define the initial one-node workflow.
    graph.add_edge(START, "interviewer")
    graph.add_edge("interviewer", END)

    return graph.compile()
