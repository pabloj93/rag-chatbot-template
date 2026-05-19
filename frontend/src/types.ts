/**
 * Domain types shared across the frontend.
 *
 * Mirrors the shapes the backend produces over the wire. Kept minimal —
 * if something can be derived from existing types, we don't redeclare it.
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

/** Conversation turn enriched with UI state (sources, streaming flag). */
export interface UIMessage extends Message {
  sources?: Source[];
  /** True while tokens are still arriving for this assistant message. */
  streaming?: boolean;
}

/**
 * Discriminated union of every SSE event the backend can emit.
 *
 * Why a union: TypeScript narrows `data` based on `event`, so the hook
 * code in useChat stays type-safe without runtime casts.
 */
export type SSEEvent =
  | { event: "sources"; data: Source[] }
  | { event: "token"; data: { text: string } }
  | { event: "done"; data: { trace_id: string; latency_ms: number; session_id: string } }
  | { event: "error"; data: { message: string } };
