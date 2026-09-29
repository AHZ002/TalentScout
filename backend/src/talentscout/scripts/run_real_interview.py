"""Run one real TalentScout interview through the current LangGraph workflow.

This script is intentionally an isolated developer harness. It reads an
existing job from PostgreSQL, runs the real retrieval/LLM services, accepts
candidate answers from the terminal, and prints the structured agent outputs.
It does not create, update, or delete database records.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from talentscout.agents.graph import build_interview_graph
from talentscout.agents.state import InterviewState
from talentscout.config.settings import get_settings
from talentscout.db.models.document import Document
from talentscout.db.models.document_chunk import DocumentChunk
from talentscout.db.models.job import Job
from talentscout.db.session import SessionFactory
from talentscout.documents.retriever import DocumentRetriever
from talentscout.embeddings.jina import JinaEmbeddingService
from talentscout.jobs.repositories.document_chunk import DocumentChunkRepository
from talentscout.llm.groq import GroqLLMService


async def get_test_job(session: AsyncSession) -> Job:
    """Select a specific job from the environment or the newest job with guidance."""
    requested_job_id = os.getenv("TALENTSCOUT_TEST_JOB_ID")

    if requested_job_id:
        job = await session.get(Job, UUID(requested_job_id))
        if job is None:
            raise RuntimeError(
                f"TALENTSCOUT_TEST_JOB_ID={requested_job_id} was not found."
            )
        return job

    # Prefer a real job that already has Additional Interview Guidance chunks,
    # because that exercises the full JD + RAG workflow rather than only JD.
    statement = (
        select(Job)
        .join(Document, Document.job_id == Job.id)
        .join(DocumentChunk, DocumentChunk.document_id == Document.id)
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    job = await session.scalar(statement)

    if job is None:
        raise RuntimeError(
            "No job with Additional Interview Guidance chunks was found. "
            "Create a job and upload guidance before running this test."
        )

    return job


def print_json(title: str, value: object) -> None:
    """Print a Pydantic model or ordinary Python value as readable JSON."""
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
    else:
        payload = value

    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


async def run() -> None:
    """Execute an interactive real-data interview session."""
    settings = get_settings()

    if not settings.groq_api_key:
        raise RuntimeError("TALENTSCOUT_GROQ_API_KEY is not configured.")
    if not settings.jina_api_key:
        raise RuntimeError("TALENTSCOUT_JINA_API_KEY is not configured.")

    async with SessionFactory() as session:
        job = await get_test_job(session)

        # Keep one database session alive for the entire harness because the
        # retriever repository uses this session for every vector-search call.
        repository = DocumentChunkRepository(session)
        embedding_service = JinaEmbeddingService()
        retriever = DocumentRetriever(repository, embedding_service)
        llm = GroqLLMService()
        graph = build_interview_graph(retriever=retriever, llm=llm)

        state: InterviewState = {
            "job_id": job.id,
            "job_description": job.description,
            "interview_history": [],
            "candidate_answer": "",
        }

        print("\nTalentScout real-data workflow test")
        print(f"Job: {job.title}")
        print(f"Job ID: {job.id}")
        print("Database writes: none")
        print("\nGenerating the first question...\n")

        state = cast(InterviewState, await graph.ainvoke(state))

        print_json("ROLE & COMPETENCY ANALYSIS", state["role_competency_analysis"])
        print_json(
            "INITIAL RETRIEVED GUIDANCE",
            state.get("retrieved_context", []),
        )
        print(f"\nINTERVIEWER QUESTION\n{state['current_question']}\n")

        while not state.get("interview_complete", False):
            answer = input("Your candidate answer (type 'quit' to stop): ").strip()

            if answer.lower() == "quit":
                print("\nTest stopped by user. No database changes were made.")
                break

            if not answer:
                print("Please enter a non-empty answer.")
                continue

            # Submit exactly one candidate answer back into the same workflow
            # state. The graph evaluates it and either asks the next question
            # or routes directly to the assessment report.
            state["candidate_answer"] = answer
            state = cast(InterviewState, await graph.ainvoke(state))

            print_json(
                f"TURN {len(state.get('interview_history', []))} EVALUATION",
                state["answer_evaluation"],
            )

            print(
                "\nInterview progress: "
                f"{len(state.get('assessed_competencies', []))} assessed, "
                f"{len(state.get('remaining_competencies', []))} remaining"
            )

            if state.get("interview_complete", False):
                print("\nThe workflow decided that enough evidence has been collected.")
                print_json("FINAL ASSESSMENT REPORT", state["assessment_report"])
                break

            print(f"\nNEXT INTERVIEW QUESTION\n{state['current_question']}\n")

        await llm.client.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        # Psycopg async connections require the Windows Selector event loop.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run())