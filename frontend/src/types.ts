/**
 * Domain types shared across the frontend.
 */

/** One turn in the conversation, as the backend's /chat schema expects. */
export interface Message {
  role: "user" | "assistant";
  content: string;
}

/** A retrieved document, as emitted in the SSE `sources` event. */
export interface Source {
  url: string;
  snippet: string;
}

/** Conversation turn enriched with UI state. */
export interface UIMessage extends Message {
  sources?: Source[];
  streaming?: boolean;
  /** Unix timestamp (ms) — used to render the per-message time badge. */
  createdAt?: number;
}

/**
 * Discriminated union of every SSE event the backend can emit.
 * TypeScript narrows `data` based on `event` — stays type-safe without casts.
 */
export type SSEEvent =
  | { event: "sources"; data: Source[] }
  | { event: "token"; data: { text: string } }
  | { event: "done"; data: { trace_id: string; latency_ms: number; session_id: string } }
  | { event: "error"; data: { message: string } };

/**
 * Progressive status of the chat pipeline.
 *   idle       → nothing in flight
 *   retrieving → request sent, waiting for sources (BM25 + Pinecone + rerank)
 *   generating → sources received, Claude is streaming tokens
 */
export type ChatStatus = "idle" | "retrieving" | "generating";

/**
 * A persisted conversation stored in localStorage.
 * `title` is the first user question (truncated to 50 chars).
 */
export interface Conversation {
  id: string;
  title: string;
  messages: UIMessage[];
  createdAt: number;
}
