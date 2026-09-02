"""LangGraph workflow for generating context-aware technical interview questions."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from talentscout.agents.state import InterviewState
from talentscout.documents.retriever import DocumentRetriever
from talentscout.llm.service import LLMService


class InterviewerAgent:
    """Generate an interview question using job and document context."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm: LLMService,
    ) -> None:
        """Initialize the interviewer with retrieval and LLM services."""
        # Retrieval provides relevant company-document information.
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

        # For the first question, retrieve documents using the JD.
        # For later questions, use the JD together with the candidate's
        # previous answer so retrieval remains relevant to the job and
        # adapts to what the candidate has already discussed.
        if candidate_answer:
            retrieval_query = (
                f"Job requirements:\n{job_description}\n\n"
                f"Candidate's previous answer:\n{candidate_answer}"
            )
        else:
            retrieval_query = job_description

        chunks = await self.retriever.retrieve(
            job_id=job_id,
            query=retrieval_query,
            limit=5,
        )

        # The LLM receives document text, not the embedding vectors.
        retrieved_context = [chunk.text for chunk in chunks]

        prompt = self._build_prompt(
            job_description=job_description,
            company_context=state.get("company_context"),
            retrieved_context=retrieved_context,
            candidate_answer=candidate_answer,
        )

        # Generate the question through the provider-independent LLM service.
        question = await self.llm.generate(
            system_prompt=(
                "You are TalentScout's technical interviewer. "
                "Generate one clear technical interview question. "
                "Base the question on the job description, optional company "
                "context, and relevant retrieved document context. "
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
        }

    @staticmethod
    def _build_prompt(
        job_description: str,
        company_context: str | None,
        retrieved_context: list[str],
        candidate_answer: str,
    ) -> str:
        """Build the interviewer prompt from all available interview context."""
        # Combine retrieved document chunks into a single context section.
        document_text = "\n\n".join(retrieved_context)

        return (
            f"Job description:\n{job_description}\n\n"
            f"Company/project context:\n"
            f"{company_context or 'No additional company context provided.'}\n\n"
            f"Relevant document context:\n"
            f"{document_text or 'No relevant document context found.'}\n\n"
            f"Candidate's previous answer:\n"
            f"{candidate_answer or 'No previous answer; this is the first question.'}\n\n"
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

    # The interviewer retrieves relevant context and generates a question.
    graph.add_node(
        "interviewer",
        interviewer.generate_question,
    )

    # Define the initial one-node workflow.
    graph.add_edge(START, "interviewer")
    graph.add_edge("interviewer", END)

    return graph.compile()
