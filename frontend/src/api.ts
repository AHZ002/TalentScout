
/**
 * Typed API client for the TalentScout FastAPI backend.
 * Keeps HTTP configuration and response validation in one place.
 */

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

// -----------------------------------------------------------------------------
// API response types
// -----------------------------------------------------------------------------

export interface Job {
  id: string;
  title: string;
  description: string;
  status: "draft" | "active" | "closed";
}

export interface GuidanceDocument {
  id: string;
  job_id: string;
  filename: string;
  content_type: string;
  storage_path: string;
  status: "processing" | "completed" | "failed";
}

export interface InterviewSession {
  id: string;
  job_id: string;
  status: string;
  current_question: string | null;
  interview_complete: boolean;
  turn_count: number;
  assessment_report: unknown | null;
}

// -----------------------------------------------------------------------------
// API error types
// -----------------------------------------------------------------------------

export interface ApiErrorDetail {
  code: string;
  message: string;
  retryable: boolean;
}

export class ApiError extends Error {
  readonly status: number | null;
  readonly code: string;
  readonly retryable: boolean;

  constructor(
    message: string,
    options: {
      status: number | null;
      code: string;
      retryable: boolean;
    },
  ) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.code = options.code;
    this.retryable = options.retryable;
  }
}

// -----------------------------------------------------------------------------
// HTTP request handler
// -----------------------------------------------------------------------------

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      `Cannot reach TalentScout API at ${API_BASE_URL}. Check that the backend is running.`,
      {
        status: null,
        code: "api_unreachable",
        retryable: true,
      },
    );
  }

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    let code = "api_error";
    let retryable = false;

    try {
      const body: {
        detail?: string | { msg?: string }[] | ApiErrorDetail;
      } = await response.json();

      if (typeof body.detail === "string") {
        message = body.detail;
      } else if (Array.isArray(body.detail)) {
        message = body.detail
          .map((item) => item.msg ?? "Invalid input")
          .join("; ");
        code = "validation_error";
      } else if (
        body.detail &&
        typeof body.detail === "object" &&
        "message" in body.detail
      ) {
        message = body.detail.message;
        code = body.detail.code;
        retryable = body.detail.retryable;
      }
    } catch {
      // Keep the HTTP status message when the server response isn't JSON.
    }

    throw new ApiError(message, {
      status: response.status,
      code,
      retryable,
    });
  }

  return (await response.json()) as T;
}

// -----------------------------------------------------------------------------
// API methods
// -----------------------------------------------------------------------------

export const api = {
  health: () => request<{ status: string }>("/health"),

  listJobs: () => request<Job[]>("/jobs"),

  createJob: (payload: { title: string; description: string }) =>
    request<Job>("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  listGuidanceDocuments: (jobId: string) =>
    request<GuidanceDocument[]>(
      `/jobs/${jobId}/additional-interview-guidance`,
    ),

  uploadGuidanceDocument: (jobId: string, file: File) => {
    const body = new FormData();
    body.append("file", file);

    return request<GuidanceDocument>(
      `/jobs/${jobId}/additional-interview-guidance`,
      { method: "POST", body },
    );
  },

  startInterview: (jobId: string) =>
    request<InterviewSession>(`/jobs/${jobId}/interviews`, {
      method: "POST",
    }),

  getInterview: (sessionId: string) =>
    request<InterviewSession>(`/interviews/${sessionId}`),

  submitAnswer: (sessionId: string, candidateAnswer: string) =>
    request<InterviewSession>(`/interviews/${sessionId}/answers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate_answer: candidateAnswer }),
    }),
};
