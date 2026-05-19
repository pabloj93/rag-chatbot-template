/**
 * RAG Chatbot — Anthropic Docs.
 *
 * Intentionally minimal demo shell — the interesting code lives in the
 * backend (rag_chain, ingest, tracing) and in useChat (SSE consumer).
 * This component is just rendering: a header, a list of messages, and
 * an input box.
 */

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { useChat } from "./useChat";

export default function App() {
  const { messages, sendMessage, isStreaming, error } = useChat();
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the latest message as the conversation grows.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;
    sendMessage(input);
    setInput("");
  }

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto p-4">
      <header className="mb-4">
        <h1 className="text-2xl font-bold">RAG Chatbot</h1>
        <p className="text-sm text-gray-600">
          Answers grounded in the Anthropic / Claude documentation.
        </p>
      </header>

      <main className="flex-1 overflow-y-auto space-y-4 mb-4">
        {messages.length === 0 && (
          <div className="text-gray-500 text-center mt-8">
            Ask something — e.g. <em>"How does prompt caching work?"</em>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={
              m.role === "user"
                ? "bg-blue-100 p-3 rounded-lg ml-12"
                : "bg-white p-3 rounded-lg mr-12 border"
            }
          >
            <div className="text-xs text-gray-500 mb-1">
              {m.role === "user" ? "You" : "Assistant"}
            </div>
            <div className="prose prose-sm max-w-none prose-pre:bg-gray-100 prose-pre:text-gray-800 prose-headings:mt-3 prose-headings:mb-1 prose-p:my-1.5">
              {m.role === "assistant" ? (
                <ReactMarkdown>{m.content}</ReactMarkdown>
              ) : (
                <p className="whitespace-pre-wrap">{m.content}</p>
              )}
              {/* Blinking cursor while tokens stream in. */}
              {m.streaming && <span className="animate-pulse">▋</span>}
            </div>
            {m.sources && m.sources.length > 0 && (
              <details className="mt-2 text-sm">
                <summary className="cursor-pointer text-gray-600">
                  {m.sources.length} sources
                </summary>
                <ul className="mt-2 space-y-2 pl-4">
                  {m.sources.map((s, j) => (
                    <li key={j}>
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-blue-600 hover:underline break-all"
                      >
                        {s.url.replace("https://platform.claude.com/docs/en/", "")}
                      </a>
                      <div className="text-xs text-gray-500 line-clamp-2">
                        {s.snippet}
                      </div>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        ))}

        {error && (
          <div className="bg-red-100 text-red-700 p-3 rounded-lg">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about Claude..."
          disabled={isStreaming}
          className="flex-1 px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
        />
        <button
          type="submit"
          disabled={isStreaming || !input.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg disabled:bg-gray-300"
        >
          {isStreaming ? "..." : "Send"}
        </button>
      </form>
    </div>
  );
}
