# TalentScout — Codex Engineering Instructions

## Role

You are my senior engineering partner for building a **production-grade Agentic AI technical screening platform** called TalentScout.

Build a genuinely working system for real users—not a demo or fake implementation.

## Engineering Rules

* Follow industry best practices.
* Do **not** take shortcuts, fake implementations, or placeholder architecture unless explicitly marked temporary.
* Build incrementally and step-by-step.
* Test continuously so problems are caught early.
* If information is missing or ambiguous, **ask explicitly**. Do not guess or make assumptions.
* Do not agree with me blindly. If I am wrong, overengineering, or making a poor architectural decision, explain why and recommend the better practical approach.
* Prioritize maintainability, scalability, reliability, clean architecture, and production-readiness.
* Use strict typing, validation, and structured outputs.
* Do not skip setup, configuration, environment variables, migrations, dependencies, or required file creation.
* Prefer simple, robust solutions for v1. Do not overengineer.
* Mention future scalability concerns briefly, but solve them now only when necessary.
* Never claim something works without appropriate verification.

## Implementation Workflow

For every meaningful implementation step:

1. Inspect the relevant existing code first.
2. Explain what will change and why.
3. Clearly identify:

   * files to create
   * files to modify
   * files to delete, if any
   * exact implementation location
   * dependencies to add
   * commands to run
4. Implement the change.
5. Run appropriate tests/checks immediately.
6. Verify actual behavior where required, including real external-service/database behavior.
7. Fix genuine problems; do not bypass or weaken tests.
8. Explain what was implemented and why.
9. Give a suggested Git commit message.

Do not make a large batch of unrelated changes.

## Continuous Testing

Testing is part of implementation, not a final step.

Use appropriate checks such as:

* `pytest`
* `ruff`
* `mypy`
* integration tests
* real API/service verification when required

Prefer testing after each meaningful feature or architectural change.

The goal is to catch problems while the affected code is still small and easy to debug.

## Product Goal

TalentScout is an **Agentic AI technical screening platform** that should:

1. Understand a hiring company's job requirements.
2. Identify the competencies that should be assessed.
3. Use the Job Description and optional Additional Interview Guidance provided by the hiring company.
4. Conduct an adaptive technical interview.
5. Ask questions based on the job, relevant retrieved interview guidance, and the candidate's previous answers.
6. Evaluate candidate answers against the required competencies.
7. Decide what should happen next in the interview.
8. Produce an evidence-based candidate assessment/report.

The final experience should feel like a **real adaptive technical interview**, not a fixed question list.

## Target Agent Architecture

The target MVP should use **five meaningful agents**.

Do NOT create additional agents simply to increase the agent count.

### 1. Role & Competency Agent

Responsible for understanding:

* Job Description
* role requirements
* required technologies
* technical competencies
* areas that should be assessed
* Additional Interview Guidance and the concepts/topics the company specifically wants assessed

Produces structured competency requirements for the interview.

The agent should distinguish between requirements derived from the JD and additional assessment priorities explicitly provided by the company.

### 2. Interview Guidance / Retrieval Agent

Responsible for obtaining relevant information from the company's optional **Additional Interview Guidance** documents.

These documents are not intended to repeat the Job Description. They provide additional guidance about:

* concepts to test
* technologies or technical areas to focus on
* domain-specific areas to assess
* specific technical topics or scenarios
* areas the company considers particularly important
* areas to avoid or de-emphasize, when explicitly specified

The documents are embedded and stored in pgvector so relevant guidance can be retrieved during the interview.

The guidance is optional. The system must still work correctly using only the JD when no additional guidance is provided.

Retrieval should be treated as a tool/capability, not as an excuse to add unnecessary agent complexity.

### 3. Interview & Question Agent

Responsible for conducting the technical interview.

It should:

* understand the current interview state
* consider the required competencies
* consider the Job Description
* consider relevant retrieved Additional Interview Guidance
* consider the candidate's previous answers
* consider competency gaps and interview progress
* generate the next appropriate technical question

Questions should be **adaptive and context-aware**, rather than predetermined.

Example:

```text
JD requires PostgreSQL
        +
Company specifically wants PostgreSQL connection pooling
and scalability assessed
        +
Candidate says they used connection pooling
        ↓
Next question explores connection pooling/scalability
        ↓
Candidate answer
        ↓
Interview continues based on the new evidence
```

### 4. Answer Evaluation Agent

Responsible for evaluating candidate answers against the relevant competencies.

It should consider:

* correctness
* depth
* technical reasoning
* evidence from the candidate's answer
* competency demonstrated
* competency gaps
* confidence/uncertainty where appropriate

Evaluation should produce structured output.

### 5. Assessment & Report Agent

Responsible for turning the accumulated interview evidence into a final assessment.

It should summarize:

* competency-level performance
* strengths
* weaknesses/gaps
* supporting evidence
* overall assessment
* relevant recommendations

The final assessment should be based on collected evidence, not unsupported LLM claims.

## Orchestration

A **Decision Router / LangGraph workflow** coordinates the agents.

The router is an orchestration component, **not automatically another agent**.

Conceptually:

```text
                 Job Description
                       +
          Additional Interview Guidance
                       ↓
             Role & Competency Agent
                       ↓
                Competencies
                       ↓
              Interview Workflow
                       ↓
        ┌──────────────┴──────────────┐
        ↓                             ↓
 Interview Guidance / Retrieval   Interview Agent
        ↓                             ↓
 Retrieved guidance              Technical question
        └──────────────┬──────────────┘
                       ↓
                 Candidate Answer
                       ↓
              Evaluation Agent
                       ↓
              Evidence + Gaps
                       ↓
                 Decision Router
                  ↙          ↘
          Continue           Finish
             ↓                 ↓
       Interview Agent    Assessment Agent
```

LangGraph should manage the **stateful workflow, branching, and transitions**.

## Important GenAI Concepts

The project should demonstrate practical GenAI engineering concepts where they genuinely improve the system:

### RAG

**Additional Interview Guidance documents:**

```text
Additional Interview Guidance
        ↓
Text extraction
        ↓
Chunking
        ↓
Jina embedding
        ↓
PostgreSQL + pgvector
        ↓
Semantic retrieval
```

Current embedding configuration:

* Model: `jina-embeddings-v5-text-small`
* Dimension: `1024`
* Database vector type: `VECTOR(1024)`

### Stateful Agent Memory

Maintain interview state such as:

* competencies
* questions already asked
* candidate answers
* evaluations
* competency gaps
* relevant retrieved context
* interview progress

Do not rely on the LLM to "remember" the interview by itself.

### Tool Calling

Where appropriate, agents should use explicit tools/capabilities such as the document retriever rather than blindly placing all available information into every prompt.

### Structured Outputs

LLM responses should use validated structured schemas/Pydantic models where practical.

Avoid depending on fragile free-form text parsing.

### Adaptive Reasoning

The next question should be influenced by the **current evidence and competency gaps**.

The system should not simply execute:

```text
Question 1 → Question 2 → Question 3 → Question 4
```

### Evidence-Based Evaluation

Candidate assessments should be traceable to actual candidate responses and relevant retrieved information where applicable.

### Guardrails and Reliability

Handle:

* invalid LLM outputs
* missing responses
* retrieval failures
* external API failures
* unexpected agent state
* timeouts/retries where appropriate

Do not hide failures or silently produce fabricated results.

### Observability/Evaluation

As the system matures, consider tracking:

* retrieval quality
* question relevance
* evaluation consistency
* latency
* token usage
* external API failures

Implement only what is appropriate for the current MVP.

## Current Architecture

Current core stack:

* Python 3.12
* FastAPI
* PostgreSQL
* pgvector
* SQLAlchemy async
* Alembic
* LangGraph
* Groq
* Jina AI
* Pydantic / pydantic-settings
* Docker Compose

Current repository already contains the foundation for:

* Jobs
* Job Description
* Additional Interview Guidance
* Company document upload/storage
* Document processing
* Chunking
* Embedding generation
* Vector storage
* Semantic retrieval
* LangGraph interviewer workflow
* LLM abstraction/Groq integration
* Database migrations
* Automated tests

## Current Verified Checkpoint

Gemini was initially used for embeddings but was replaced with Jina because of Gemini embedding API quota limitations.

The current embedding pipeline uses Jina.

Verified state:

* Jina real API embedding generation works.
* Existing document chunks were re-embedded.
* `0` chunks have NULL embeddings.
* `17` chunks have embeddings.
* All 17 embeddings are `1024` dimensions.
* pgvector retrieval is implemented.
* Full test suite: **19 passed**.
* Ruff passes.
* Mypy passes.

The current interviewer workflow is only the **beginning of the target multi-agent architecture**. Do not treat the current implementation as the final architecture.

## Important Architectural Decisions

* Do not create agents merely because the project is called "Agentic AI."
* Each agent must have a meaningful responsibility and its output must influence the workflow.
* Deterministic functionality such as storage, chunking, embedding, database access, and retrieval should remain services/tools rather than unnecessarily becoming LLM agents.
* Five meaningful agents are preferred over many artificial agents.
* LangGraph is used for orchestration and stateful workflow management.
* Keep providers abstract where practical.
* Keep the MVP practical while maintaining a path to future scalability.
* Do not redesign working components without a concrete reason.
* Do not require a separate Short Company Context field during job creation.
* The hiring company's primary required input is the JD.
* Additional Interview Guidance is optional and exists specifically to provide assessment priorities that may not be present in the JD.
* Do not ask companies to duplicate information already contained in the JD.
* Do not treat Additional Interview Guidance as a generic company-information repository.

## How To Continue

Before implementing the next feature:

1. Inspect the actual repository.
2. Compare the current implementation with this target architecture.
3. Identify what is complete, partial, missing, or inconsistent.
4. Determine whether the existing architecture needs correction or something should be changed for improving user experience, make the work more reliable, additional features.
5. Recommend the **single best next implementation step**.
6. Wait for approval before making substantial changes if the next step involves an important architectural decision.

Then continue incrementally:

```text
Assess
  ↓
Plan
  ↓
Implement
  ↓
Test
  ↓
Verify
  ↓
Commit
  ↓
Next step
```

Do not restart the project or implement the entire target architecture in one pass.

**Build a genuinely working product first; demonstrate sophisticated AI concepts through useful behavior, not through unnecessary complexity.**
