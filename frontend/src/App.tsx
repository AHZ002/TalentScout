
/**
 * TalentScout MVP interface.
 * Provides recruiter job management, guidance-document uploads,
 * and a candidate-facing adaptive interview experience.
 */

import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api, type GuidanceDocument, type InterviewSession, type Job } from "./api";
import "./App.css";

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}


function InterviewView({ sessionId }: { sessionId: string }) {
  const [session, setSession] = useState<InterviewSession | null>(null);
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const result = await api.getInterview(sessionId);
      setSession(result);
      setError("");
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

useEffect(() => {
  let cancelled = false;

  api
    .getInterview(sessionId)
    .then((result) => {
      if (!cancelled) {
        setSession(result);
        setError("");
      }
    })
    .catch((cause: unknown) => {
      if (!cancelled) {
        setError(errorMessage(cause));
      }
    })
    .finally(() => {
      if (!cancelled) {
        setLoading(false);
      }
    });

  return () => {
    cancelled = true;
  };
}, [sessionId]);

  async function submitAnswer(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!answer.trim() || submitting) return;

    setSubmitting(true);
    setError("");
    try {
      const updated = await api.submitAnswer(sessionId, answer.trim());
      setSession(updated);
      setAnswer("");
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="candidate-shell">
      <header className="topbar">
        <a className="brand" href="#/">TalentScout</a>
        <span className="muted">Technical screening</span>
      </header>

      <section className="panel interview-panel">
        <p className="eyebrow">CANDIDATE INTERVIEW</p>
        {loading ? <p>Loading interview…</p> : null}
        {error ? (
          <div className="alert error">
            <p>{error}</p>
            <button className="button secondary" onClick={() => void refresh()}>
              Reload interview
            </button>
          </div>
        ) : null}

        {session && !loading ? (
          <>
            <div className="progress-row">
              <span>Question {Math.max(session.turn_count, 1)}</span>
              <span className="status-pill">
                {session.interview_complete ? "Completed" : "In progress"}
              </span>
            </div>

            {session.interview_complete ? (
              <section className="completion">
                <div className="completion-mark">✓</div>
                <h1>Interview completed</h1>
                <p className="muted">
                  Your responses have been submitted. The assessment is available
                  below when the backend has generated it.
                </p>
                {session.assessment_report ? (
                  <pre className="report">
                    {JSON.stringify(session.assessment_report, null, 2)}
                  </pre>
                ) : (
                  <p className="muted">No assessment report was returned.</p>
                )}
                <a className="button secondary" href="#/">Back to TalentScout</a>
              </section>
            ) : (
              <>
                <h1 className="question-heading">Your next question</h1>
                {session.current_question ? (
                  <div className="question-card">{session.current_question}</div>
                ) : (
                  <div className="alert error">
                    The interview has no current question. Please contact the recruiter.
                  </div>
                )}

                <form className="answer-form" onSubmit={submitAnswer}>
                  <label htmlFor="candidate-answer">Your answer</label>
                  <textarea
                    id="candidate-answer"
                    value={answer}
                    onChange={(event) => setAnswer(event.target.value)}
                    placeholder="Explain your approach, reasoning, and relevant examples…"
                    rows={7}
                    required
                    minLength={1}
                    disabled={submitting || !session.current_question}
                  />
                  <div className="form-footer">
                    <span className="muted">
                      Be specific and explain your reasoning.
                    </span>
                    <button
                      className="button primary"
                      type="submit"
                      disabled={submitting || !answer.trim() || !session.current_question}
                    >
                      {submitting ? "Evaluating…" : "Submit answer"}
                    </button>
                  </div>
                </form>
              </>
            )}
          </>
        ) : null}
      </section>
    </main>
  );
}

function RecruiterDashboard() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [documents, setDocuments] = useState<GuidanceDocument[]>([]);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const selectedJob = jobs.find((job) => job.id === selectedJobId);

const loadJobs = useCallback(async () => {
  try {
    const result = await api.listJobs();
    setJobs(result);
    setSelectedJobId((current) =>
      result.some((job) => job.id === current) ? current : result[0]?.id ?? "",
    );
    setError("");
  } catch (cause) {
    setError(errorMessage(cause));
  } finally {
    setLoading(false);
  }
}, []);

  useEffect(() => {
    let cancelled = false;

    api
      .listJobs()
      .then((result) => {
        if (!cancelled) {
          setJobs(result);
          setSelectedJobId((current) =>
            result.some((job) => job.id === current) ? current : result[0]?.id ?? "",
          );
          setError("");
        }
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(errorMessage(cause));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    const request = selectedJobId
      ? api.listGuidanceDocuments(selectedJobId)
      : Promise.resolve<GuidanceDocument[]>([]);

    request
      .then((result) => {
        if (!cancelled) {
          setDocuments(result);
        }
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(errorMessage(cause));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedJobId]);

  async function createJob(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const created = await api.createJob({
        title: title.trim(),
        description: description.trim(),
      });
      setJobs((current) => [created, ...current]);
      setSelectedJobId(created.id);
      setTitle("");
      setDescription("");
      setNotice("Job created successfully.");
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function uploadDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedJobId || !file) return;

    setBusy(true);
    setError("");
    setNotice("");
    try {
      const uploaded = await api.uploadGuidanceDocument(selectedJobId, file);
      setDocuments((current) => [uploaded, ...current]);
      setFile(null);
      const input = document.getElementById("guidance-file") as HTMLInputElement | null;
      if (input) input.value = "";
      setNotice(
        uploaded.status === "completed"
          ? "Guidance uploaded and processed."
          : `Guidance upload status: ${uploaded.status}.`,
      );
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function launchInterview() {
    if (!selectedJobId) return;
    setBusy(true);
    setError("");
    try {
      const session = await api.startInterview(selectedJobId);
      window.location.hash = `/interviews/${session.id}`;
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#/">TalentScout</a>
        <span className="muted">Adaptive technical screening</span>
        <button className="button secondary small" onClick={() => void loadJobs()}>
          Refresh
        </button>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">RECRUITER WORKSPACE</p>
          <h1>Build better technical interviews.</h1>
          <p className="muted">
            Create a role, add relevant guidance, and launch an AI-led interview.
          </p>
        </div>
        <div className="hero-stat">
          <strong>{jobs.length}</strong>
          <span>Hiring {jobs.length === 1 ? "role" : "roles"}</span>
        </div>
      </section>

      {error ? (
        <div className="alert error">
          <span>{error}</span>
          <button className="text-button" onClick={() => setError("")}>Dismiss</button>
        </div>
      ) : null}
      {notice ? (
        <div className="alert success">
          <span>{notice}</span>
          <button className="text-button" onClick={() => setNotice("")}>Dismiss</button>
        </div>
      ) : null}

      <div className="dashboard-grid">
        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">YOUR WORKSPACE</p>
              <h2>Hiring roles</h2>
            </div>
            <span className="count">{jobs.length}</span>
          </div>

          {loading ? <p className="muted">Loading roles…</p> : null}
          {!loading && jobs.length === 0 ? (
            <div className="empty-state">
              <h3>No roles yet</h3>
              <p className="muted">Create your first role to start using TalentScout.</p>
            </div>
          ) : null}

          <div className="job-list">
            {jobs.map((job) => (
              <button
                key={job.id}
                className={`job-item ${selectedJobId === job.id ? "selected" : ""}`}
                onClick={() => {
                  setSelectedJobId(job.id);
                  setNotice("");
                }}
              >
                <span className="job-icon">TS</span>
                <span className="job-copy">
                  <strong>{job.title}</strong>
                  <span className="muted">{job.description}</span>
                </span>
                <span className="status-pill">{job.status}</span>
              </button>
            ))}
          </div>

          <div className="subsection">
            <h3>Create a role</h3>
            <form className="stack-form" onSubmit={createJob}>
              <label htmlFor="job-title">Job title</label>
              <input
                id="job-title"
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                placeholder="e.g. Machine Learning Engineer"
                maxLength={200}
                required
              />
              <label htmlFor="job-description">Job description</label>
              <textarea
                id="job-description"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="Responsibilities, skills, and experience required…"
                rows={4}
                required
              />
              <button className="button primary" disabled={busy}>
                {busy ? "Please wait…" : "Create role"}
              </button>
            </form>
          </div>
        </section>

        <section className="panel role-detail">
          {selectedJob ? (
            <>
              <p className="eyebrow">ROLE SETUP</p>
              <div className="detail-title">
                <div>
                  <h2>{selectedJob.title}</h2>
                  <span className="status-pill">{selectedJob.status}</span>
                </div>
              </div>
              <p className="description">{selectedJob.description}</p>

              <div className="divider" />

              <div className="section-heading">
                <div>
                  <p className="eyebrow">COMPANY KNOWLEDGE</p>
                  <h3>Additional Interview Guidance</h3>
                </div>
                <span className="count">{documents.length}</span>
              </div>
              <p className="muted">
                Upload a PDF or plain-text file with product, system, or role context.
                This information can ground interview questions.
              </p>

              <form className="upload-box" onSubmit={uploadDocument}>
                <label htmlFor="guidance-file">Choose a PDF or text file</label>
                <input
                  id="guidance-file"
                  type="file"
                  accept=".pdf,.txt,application/pdf,text/plain"
                  onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                />
                <button
                  className="button secondary"
                  type="submit"
                  disabled={busy || !file}
                >
                  {busy ? "Processing…" : "Upload guidance"}
                </button>
              </form>

              <div className="document-list">
                {documents.map((doc) => (
                  <div className="document-item" key={doc.id}>
                    <span className="document-icon">PDF</span>
                    <span className="document-copy">
                      <strong>{doc.filename}</strong>
                      <span className="muted">{doc.content_type}</span>
                    </span>
                    <span className={`status-pill ${doc.status === "failed" ? "failed" : ""}`}>
                      {doc.status}
                    </span>
                  </div>
                ))}
                {documents.length === 0 ? (
                  <p className="muted empty-docs">No guidance documents uploaded for this role.</p>
                ) : null}
              </div>

              <div className="launch-card">
                <div>
                  <h3>Ready to screen a candidate?</h3>
                  <p className="muted">
                    Start an interview and open the candidate interview view.
                  </p>
                </div>
                <button
                  className="button primary"
                  onClick={() => void launchInterview()}
                  disabled={busy || selectedJob.status === "closed"}
                >
                  {busy ? "Starting…" : "Start interview"}
                </button>
              </div>
              {selectedJob.status === "closed" ? (
                <p className="muted">Closed roles cannot be launched from this screen.</p>
              ) : null}
            </>
          ) : (
            <div className="empty-state large">
              <div className="empty-symbol">↗</div>
              <h2>Select a role</h2>
              <p className="muted">Choose an existing role or create a new one to get started.</p>
            </div>
          )}
        </section>
      </div>

      <footer className="footer">TalentScout · AI-powered technical screening</footer>
    </main>
  );
}

function App() {
  const [hash, setHash] = useState(window.location.hash);

  useEffect(() => {
    const onHashChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const match = hash.match(/^#\/interviews\/([0-9a-f-]+)$/i);

  return match ? <InterviewView sessionId={match[1]} /> : <RecruiterDashboard />;
}

export default App;
