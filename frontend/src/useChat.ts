/**
 * useChat — React hook that owns the live conversation state and consumes
 * the backend's SSE streaming endpoint.
 *
 * Public surface:
 *     const { messages, sendMessage, status, error, loadMessages } = useChat();
 *
 * status drives the progressive loading indicator in App.tsx:
 *   "retrieving" → "Searching docs..."  (pipeline is doing BM25+Pinecone+rerank)
 *   "generating" → tokens stream in     (Claude is generating)
 *   "idle"       → nothing in flight
 */

import { useCallback, useRef, useState } from "react";
import type { ChatStatus, SSEEvent, UIMessage } from "./types";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";


export function useChat() {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  // Session ID persists across turns so LangFuse groups them into one session.
  const sessionIdRef = useRef<string | null>(null);

  /**
   * Replace the current message list (used when loading a past conversation
   * from the sidebar). Resets status and error too.
   */
  function loadMessages(msgs: UIMessage[]) {
    setMessages(msgs);
    setStatus("idle");
    setError(null);
    sessionIdRef.current = null; // new session when continuing old conversation
  }

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || status !== "idle") return;

      setError(null);
      setStatus("retrieving"); // ← user sees "Searching docs..." until sources arrive

      const userMessage: UIMessage = { role: "user", content: text, createdAt: Date.now() };
      const assistantPlaceholder: UIMessage = {
        role: "assistant",
        content: "",
        streaming: true,
        createdAt: Date.now(),
      };
      const currentMessages = [...messages, userMessage];
      setMessages([...currentMessages, assistantPlaceholder]);

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
        if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

        await consumeSSE(response.body, (evt) => {
          switch (evt.event) {
            case "sources":
              // Sources received → retrieval done, Claude has started generating.
              setStatus("generating");
              setMessages((prev) => updateLast(prev, { sources: evt.data }));
              break;
            case "token":
              setMessages((prev) =>
                updateLast(prev, {
                  content: (prev[prev.length - 1]?.content ?? "") + evt.data.text,
                })
              );
              break;
            case "done":
              sessionIdRef.current = evt.data.session_id;
              setMessages((prev) => updateLast(prev, { streaming: false }));
              setStatus("idle");
              break;
            case "error":
              throw new Error(evt.data.message);
          }
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error");
        setMessages((prev) => updateLast(prev, { streaming: false }));
        setStatus("idle");
      }
    },
    [messages, status]
  );

  return { messages, sendMessage, status, error, loadMessages };
}


// --- helpers ----------------------------------------------------------------

function updateLast(messages: UIMessage[], patch: Partial<UIMessage>): UIMessage[] {
  if (messages.length === 0) return messages;
  const last = messages[messages.length - 1];
  return [...messages.slice(0, -1), { ...last, ...patch }];
}

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
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      const evt = parseSSEEvent(chunk);
      if (evt) onEvent(evt);
    }
  }
}

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
