"""Shared types used across the interview workflow."""

from typing import TypedDict

from talentscout.agents.answer_evaluation import AnswerEvaluation


class InterviewTurn(TypedDict):
    """Represent one completed question-and-answer exchange."""

    question: str
    candidate_answer: str
    evaluation: AnswerEvaluation