/**
 * useChat — React hook that owns the conversation state and consumes
 * the backend's SSE streaming endpoint.
 *
 * Public surface:
 *     const { messages, sendMessage, isStreaming, error } = useChat();
 *
 * The SSE parser is hand-rolled (no library). We read the response body
 * as a Uint8Array stream, split on the SSE `\n\n` delimiter, and dispatch
 * each event to setMessages. Keeping it manual makes the wire format
 * explicit and avoids pulling in a 5KB dependency.
 */

import { useCallback, useRef, useState } from "react";
import type { SSEEvent, UIMessage } from "./types";

// In local dev: VITE_BACKEND_URL=http://localhost:8000 (from .env)
// In docker-compose: http://localhost:8000 (from build arg)
// In HF Spaces single container: "" (not set) → fetch("/chat/stream")
//   resolves relative to the page origin, which IS the FastAPI server.
// Why ?? instead of ||: ?? falls back only on undefined/null, not on ""
//   so an explicitly-empty VITE_BACKEND_URL is preserved as "".
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";


export function useChat() {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Session ID persists across messages so LangFuse groups every turn
  // of a conversation into one session. The backend mints it on the
  // first /chat/stream call and echoes it back in the `done` event.
  const sessionIdRef = useRef<string | null>(null);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isStreaming) return;
      setError(null);
      setIsStreaming(true);

      // Push the user's turn and an empty assistant placeholder. The
      // placeholder is what fills up token-by-token as the SSE stream
      // arrives — that's what makes the UI feel "alive".
      const userMessage: UIMessage = { role: "user", content: text };
      const assistantPlaceholder: UIMessage = {
        role: "assistant",
        content: "",
        streaming: true,
      };
      const currentMessages = [...messages, userMessage];
      setMessages([...currentMessages, assistantPlaceholder]);

      // Backend expects the full transcript in `messages`, with the
      // new user turn as the last entry (see routers/chat.py).
      const body = {
        messages: currentMessages.map(({ role, content }) => ({ role, content })),
        session_id: sessionIdRef.current,
      };

      try {
        const response = await fetch(`${BACKEND_URL}/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!response.ok || !response.body) {
          throw new Error(`HTTP ${response.status}`);
        }

        await consumeSSE(response.body, (evt) => {
          switch (evt.event) {
            case "sources":
              setMessages((prev) => updateLast(prev, { sources: evt.data }));
              break;
            case "token":
              // Append the new token text to the placeholder. We compute
              // the new content inside the updater so concurrent updates
              // don't drop characters.
              setMessages((prev) =>
                updateLast(prev, {
                  content: (prev[prev.length - 1]?.content ?? "") + evt.data.text,
                })
              );
              break;
            case "done":
              sessionIdRef.current = evt.data.session_id;
              setMessages((prev) => updateLast(prev, { streaming: false }));
              break;
            case "error":
              throw new Error(evt.data.message);
          }
        });
      } catch (e) {
        const message = e instanceof Error ? e.message : "Unknown error";
        setError(message);
        // Mark the placeholder as done so the spinner stops.
        setMessages((prev) => updateLast(prev, { streaming: false }));
      } finally {
        setIsStreaming(false);
      }
    },
    [messages, isStreaming]
  );

  return { messages, sendMessage, isStreaming, error };
}


// --- helpers ---------------------------------------------------------

/** Apply a partial patch to the last message in the list (immutable). */
function updateLast(messages: UIMessage[], patch: Partial<UIMessage>): UIMessage[] {
  if (messages.length === 0) return messages;
  const last = messages[messages.length - 1];
  return [...messages.slice(0, -1), { ...last, ...patch }];
}


/**
 * Consume a ReadableStream of SSE-formatted bytes and call `onEvent` for
 * each fully-received event. ~25 lines of wire parsing — the alternative
 * is to depend on `@microsoft/fetch-event-source` or similar, which is
 * not worth it for code this small.
 */
async function consumeSSE(
  body: ReadableStream<Uint8Array>,
  onEvent: (evt: SSEEvent) => void
) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    // `stream: true` keeps the decoder happy across multi-byte UTF-8
    // boundaries that get split between chunks.
    buffer += decoder.decode(value, { stream: true });

    // SSE separates events with a blank line ("\n\n"). The last fragment
    // may be incomplete, so we keep it in the buffer for the next read.
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const evt = parseSSEEvent(chunk);
      if (evt) onEvent(evt);
    }
  }
}


/** Parse one SSE event block (lines like `event: foo` + `data: {...}`). */
function parseSSEEvent(raw: string): SSEEvent | null {
  let eventType = "";
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event: ")) eventType = line.slice(7).trim();
    else if (line.startsWith("data: ")) data = line.slice(6);
  }
  if (!eventType || !data) return null;
  try {
    return { event: eventType as SSEEvent["event"], data: JSON.parse(data) } as SSEEvent;
  } catch {
    return null;
  }
}
