import { useState, useRef, useCallback, useEffect } from "react";
import { BACKEND_URL } from "../App";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  error?: boolean;
}

let msgIdCounter = 0;
const newId = () => `msg-${++msgIdCounter}`;

export default function ChatMode() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [useReasoning, setUseReasoning] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [backendOffline, setBackendOffline] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const appendChunk = useCallback((id: string, chunk: string) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, content: m.content + chunk } : m))
    );
  }, []);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || streaming) return;

    const userMsg: Message = { id: newId(), role: "user", content: text };
    const aiId = newId();
    const aiMsg: Message = { id: aiId, role: "assistant", content: "" };

    setMessages((prev) => [...prev, userMsg, aiMsg]);
    setInput("");
    setStreaming(true);
    setBackendOffline(false);

    try {
      if (useReasoning) {
        // Non-streaming reasoning path
        const res = await fetch(`${BACKEND_URL}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: text, use_reasoning: true }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as { response?: string; answer?: string };
        const reply = data.response ?? data.answer ?? JSON.stringify(data);
        setMessages((prev) =>
          prev.map((m) => (m.id === aiId ? { ...m, content: reply } : m))
        );
      } else {
        // SSE streaming
        const es = new EventSource(
          `${BACKEND_URL}/query/stream?query=${encodeURIComponent(text)}`
        );
        esRef.current = es;

        es.onmessage = (e) => {
          const raw = e.data as string;
          if (raw === "[DONE]") {
            es.close();
            setStreaming(false);
            return;
          }
          try {
            const parsed = JSON.parse(raw) as { chunk?: string; token?: string; text?: string };
            const chunk = parsed.chunk ?? parsed.token ?? parsed.text ?? raw;
            appendChunk(aiId, chunk);
          } catch {
            appendChunk(aiId, raw);
          }
        };

        es.onerror = () => {
          es.close();
          setStreaming(false);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiId && m.content === ""
                ? { ...m, content: "Error: could not reach backend.", error: true }
                : m
            )
          );
          setBackendOffline(true);
        };

        // Fallback: also try POST stream
        es.addEventListener("done", () => {
          es.close();
          setStreaming(false);
        });
      }
    } catch (err) {
      setBackendOffline(true);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === aiId
            ? { ...m, content: `Error: ${String(err)}`, error: true }
            : m
        )
      );
      setStreaming(false);
    }
  }, [input, streaming, useReasoning, appendChunk]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  };

  const stopStreaming = () => {
    esRef.current?.close();
    setStreaming(false);
  };

  const clearChat = () => {
    stopStreaming();
    setMessages([]);
  };

  return (
    <div className="chat-mode">
      {backendOffline && (
        <div className="offline-banner">
          ⚠️ Backend unreachable at <code>{BACKEND_URL}</code>. Start the FastAPI
          server and retry.
        </div>
      )}

      <div className="message-list">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">🤖</div>
            <p>Ask WISPR anything.</p>
            <p className="empty-hint">Press Enter to send, Shift+Enter for newline.</p>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`message message-${m.role} ${m.error ? "message-error" : ""}`}>
            <div className="message-avatar">
              {m.role === "user" ? "👤" : "🤖"}
            </div>
            <div className="message-body">
              <pre className="message-content">{m.content || (streaming ? "▌" : "")}</pre>
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <div className="chat-controls">
          <label className="reasoning-toggle">
            <input
              type="checkbox"
              checked={useReasoning}
              onChange={(e) => setUseReasoning(e.target.checked)}
            />
            Use Reasoning
          </label>
          <button className="btn-ghost" onClick={clearChat} title="Clear chat">
            🗑 Clear
          </button>
        </div>
        <div className="input-row">
          <textarea
            className="chat-textarea"
            rows={3}
            placeholder="Type your message… (Enter to send)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={streaming}
          />
          {streaming ? (
            <button className="btn-stop" onClick={stopStreaming}>
              ⏹ Stop
            </button>
          ) : (
            <button
              className="btn-send"
              onClick={() => void sendMessage()}
              disabled={!input.trim()}
            >
              Send ▶
            </button>
          )}
        </div>
      </div>

      <style>{`
        .chat-mode {
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
        }
        .offline-banner {
          background: #3a1a1a;
          color: #f87171;
          padding: 8px 16px;
          font-size: 0.82rem;
          border-bottom: 1px solid #5a2020;
        }
        .offline-banner code {
          color: #fbbf24;
        }
        .message-list {
          flex: 1;
          overflow-y: auto;
          padding: 16px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .empty-state {
          margin: auto;
          text-align: center;
          color: #555;
          user-select: none;
        }
        .empty-icon { font-size: 2.5rem; margin-bottom: 8px; }
        .empty-hint { font-size: 0.78rem; margin-top: 4px; }
        .message {
          display: flex;
          gap: 10px;
          align-items: flex-start;
          max-width: 820px;
        }
        .message-user { align-self: flex-end; flex-direction: row-reverse; }
        .message-avatar { font-size: 1.2rem; flex-shrink: 0; padding-top: 2px; }
        .message-body {
          background: #1e1e30;
          border-radius: 10px;
          padding: 10px 14px;
          border: 1px solid #2d2d50;
          max-width: 700px;
        }
        .message-user .message-body {
          background: #1d2540;
          border-color: #3040a0;
        }
        .message-error .message-body { border-color: #7f2020; background: #2a1010; }
        .message-content {
          white-space: pre-wrap;
          word-break: break-word;
          font-family: inherit;
          font-size: 0.9rem;
          line-height: 1.6;
          color: #ddd;
          margin: 0;
        }
        .chat-input-area {
          border-top: 1px solid #222;
          padding: 10px 16px 14px;
          background: #12121f;
          flex-shrink: 0;
        }
        .chat-controls {
          display: flex;
          align-items: center;
          gap: 14px;
          margin-bottom: 8px;
        }
        .reasoning-toggle {
          display: flex;
          align-items: center;
          gap: 6px;
          cursor: pointer;
          color: #aaa;
          font-size: 0.82rem;
        }
        .reasoning-toggle input { width: 14px; height: 14px; cursor: pointer; }
        .btn-ghost {
          background: transparent;
          border: 1px solid #333;
          color: #888;
          padding: 4px 10px;
          border-radius: 5px;
          cursor: pointer;
          font-size: 0.78rem;
        }
        .btn-ghost:hover { border-color: #555; color: #bbb; }
        .input-row { display: flex; gap: 8px; align-items: flex-end; }
        .chat-textarea {
          flex: 1;
          resize: none;
          border-radius: 8px;
          background: #1a1a2e;
          border: 1px solid #333;
          color: #e0e0e0;
          padding: 10px 12px;
          font-size: 0.9rem;
          line-height: 1.5;
        }
        .chat-textarea:disabled { opacity: 0.6; }
        .btn-send, .btn-stop {
          padding: 8px 18px;
          border-radius: 8px;
          border: none;
          font-size: 0.88rem;
          cursor: pointer;
          white-space: nowrap;
          height: fit-content;
        }
        .btn-send {
          background: #7c6af7;
          color: #fff;
        }
        .btn-send:hover:not(:disabled) { background: #6a58e0; }
        .btn-send:disabled { opacity: 0.45; cursor: not-allowed; }
        .btn-stop { background: #8b2020; color: #fff; }
        .btn-stop:hover { background: #a02525; }
      `}</style>
    </div>
  );
}
