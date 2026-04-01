import { useState, useRef, useCallback, useEffect } from "react";
import { open as tauriOpen } from "@tauri-apps/api/dialog";
import { open as shellOpen } from "@tauri-apps/api/shell";
import { BACKEND_URL } from "../App";
import StreamLog, { LogEntry } from "./StreamLog";

// ─── Types ────────────────────────────────────────────────────────────────────

type CodeEngineMode = "ask" | "architect" | "code" | "debug" | "orchestrator";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  error?: boolean;
}

interface CompileResult {
  zip_path?: string;
  exe_path?: string;
  output_dir?: string;
  success?: boolean;
  message?: string;
}

let idCounter = 0;
const newId = () => `ce-${++idCounter}`;

// ─── Component ────────────────────────────────────────────────────────────────

export default function CodingMode() {
  // --- Code Engine Chat state ---
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [sessionId, setSessionId] = useState<string>("");
  const [ceMode, setCeMode] = useState<CodeEngineMode>("code");
  const [workspace, setWorkspace] = useState<string>("");
  const [ceChatStreaming, setCeChatStreaming] = useState(false);
  const ceChatEsRef = useRef<EventSource | null>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // --- Xencode Compile state ---
  const [xcWorkspace, setXcWorkspace] = useState<string>("");
  const [xcSpec, setXcSpec] = useState<string>("");
  const [xcStreaming, setXcStreaming] = useState(false);
  const [xcLogs, setXcLogs] = useState<LogEntry[]>([]);
  const [xcResult, setXcResult] = useState<CompileResult | null>(null);
  const [xcError, setXcError] = useState<string>("");
  const xcEsRef = useRef<EventSource | null>(null);

  // --- GitHub publish state ---
  const [publishRepoName, setPublishRepoName] = useState<string>("");
  const [publishPrivate, setPublishPrivate] = useState(true);
  const [publishing, setPublishing] = useState(false);
  const [publishResult, setPublishResult] = useState<string>("");

  // --- Backend offline ---
  const [backendOffline, setBackendOffline] = useState(false);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  // ── Code Engine Chat ─────────────────────────────────────────────────────

  const appendCeChunk = useCallback((id: string, chunk: string) => {
    setChatMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, content: m.content + chunk } : m))
    );
  }, []);

  const sendCeChat = useCallback(async () => {
    const text = chatInput.trim();
    if (!text || ceChatStreaming) return;

    const userMsg: ChatMessage = { id: newId(), role: "user", content: text };
    const aiId = newId();
    const aiMsg: ChatMessage = { id: aiId, role: "assistant", content: "" };

    setChatMessages((prev) => [...prev, userMsg, aiMsg]);
    setChatInput("");
    setCeChatStreaming(true);
    setBackendOffline(false);

    const params = new URLSearchParams({
      message: text,
      mode: ceMode,
      ...(sessionId ? { session_id: sessionId } : {}),
      ...(workspace ? { workspace_path: workspace } : {}),
    });

    const es = new EventSource(`${BACKEND_URL}/code-engine/chat/stream?${params}`);
    ceChatEsRef.current = es;

    es.onmessage = (e) => {
      const raw = e.data as string;
      if (raw === "[DONE]") {
        es.close();
        setCeChatStreaming(false);
        return;
      }
      try {
        const parsed = JSON.parse(raw) as {
          chunk?: string;
          token?: string;
          session_id?: string;
          content?: string;
        };
        if (parsed.session_id && !sessionId) setSessionId(parsed.session_id);
        const chunk = parsed.chunk ?? parsed.token ?? parsed.content ?? raw;
        appendCeChunk(aiId, chunk);
      } catch {
        appendCeChunk(aiId, raw);
      }
    };

    es.addEventListener("done", () => {
      es.close();
      setCeChatStreaming(false);
    });

    es.onerror = () => {
      es.close();
      setCeChatStreaming(false);
      setChatMessages((prev) =>
        prev.map((m) =>
          m.id === aiId && m.content === ""
            ? { ...m, content: "Error: backend unreachable.", error: true }
            : m
        )
      );
      setBackendOffline(true);
    };
  }, [chatInput, ceChatStreaming, ceMode, sessionId, workspace, appendCeChunk]);

  const handleCeChatKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendCeChat();
    }
  };

  const newCeSession = () => {
    ceChatEsRef.current?.close();
    setCeChatStreaming(false);
    setChatMessages([]);
    setSessionId("");
  };

  // ── Workspace picker (Tauri dialog) ──────────────────────────────────────

  const pickWorkspace = useCallback(async (setter: (v: string) => void) => {
    try {
      const selected = await tauriOpen({ directory: true, multiple: false });
      if (typeof selected === "string") setter(selected);
    } catch {
      // Running in browser dev mode – prompt fallback
      const val = prompt("Enter workspace path:");
      if (val) setter(val);
    }
  }, []);

  // ── Xencode Compile ──────────────────────────────────────────────────────

  const addLog = useCallback((entry: LogEntry) => {
    setXcLogs((prev) => [...prev, entry]);
  }, []);

  const runXencode = useCallback(async () => {
    const spec = xcSpec.trim();
    if (!spec || xcStreaming) return;

    setXcStreaming(true);
    setXcLogs([]);
    setXcResult(null);
    setXcError("");
    setBackendOffline(false);

    const params = new URLSearchParams({
      spec,
      ...(xcWorkspace ? { workspace_path: xcWorkspace } : {}),
    });

    const es = new EventSource(
      `${BACKEND_URL}/code-engine/xencode/compile/stream?${params}`
    );
    xcEsRef.current = es;

    es.onmessage = (e) => {
      const raw = e.data as string;
      if (raw === "[DONE]") {
        es.close();
        setXcStreaming(false);
        return;
      }
      try {
        const parsed = JSON.parse(raw) as {
          event?: string;
          type?: string;
          data?: unknown;
          message?: string;
          zip_path?: string;
          exe_path?: string;
          output_dir?: string;
        };
        const evtType = parsed.event ?? parsed.type ?? "info";

        if (evtType === "done" || evtType === "complete") {
          setXcResult({
            zip_path: parsed.zip_path,
            exe_path: parsed.exe_path,
            output_dir: parsed.output_dir,
            success: true,
          });
          es.close();
          setXcStreaming(false);
        } else if (evtType === "error") {
          setXcError(parsed.message ?? "Unknown error");
          es.close();
          setXcStreaming(false);
        } else {
          addLog({
            type: evtType,
            message:
              typeof parsed.data === "string"
                ? parsed.data
                : (parsed.message ?? JSON.stringify(parsed)),
          });
        }
      } catch {
        addLog({ type: "info", message: raw });
      }
    };

    es.addEventListener("done", () => {
      es.close();
      setXcStreaming(false);
    });

    es.onerror = () => {
      es.close();
      setXcStreaming(false);
      setXcError("Backend unreachable. Start the FastAPI server.");
      setBackendOffline(true);
    };
  }, [xcSpec, xcWorkspace, xcStreaming, addLog]);

  // ── GitHub Publish ────────────────────────────────────────────────────────

  const publishToGitHub = useCallback(async () => {
    if (!publishRepoName.trim() || !xcResult?.output_dir) return;
    setPublishing(true);
    setPublishResult("");

    try {
      const res = await fetch(`${BACKEND_URL}/code-engine/github/create-and-publish`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo_name: publishRepoName.trim(),
          private: publishPrivate,
          source_dir: xcResult.output_dir,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as { url?: string; message?: string };
      setPublishResult(data.url ?? data.message ?? "Published!");
    } catch (err) {
      setPublishResult(`Error: ${String(err)}`);
    } finally {
      setPublishing(false);
    }
  }, [publishRepoName, publishPrivate, xcResult]);

  const openInExplorer = useCallback(async (path: string) => {
    try {
      await shellOpen(path);
    } catch {
      alert(`Open manually: ${path}`);
    }
  }, []);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="coding-mode">
      {backendOffline && (
        <div className="offline-banner">
          ⚠️ Backend unreachable at <code>{BACKEND_URL}</code>
        </div>
      )}

      {/* ── Top: Code Engine Chat ── */}
      <section className="ce-section">
        <div className="section-header">
          <span className="section-title">⚙️ Code Engine Chat</span>
          <div className="ce-header-controls">
            <label>Mode</label>
            <select
              value={ceMode}
              onChange={(e) => setCeMode(e.target.value as CodeEngineMode)}
            >
              {(["ask", "architect", "code", "debug", "orchestrator"] as CodeEngineMode[]).map(
                (m) => <option key={m} value={m}>{m}</option>
              )}
            </select>
            <label>Workspace</label>
            <div className="path-pick">
              <input
                type="text"
                placeholder="/path/to/workspace"
                value={workspace}
                onChange={(e) => setWorkspace(e.target.value)}
                style={{ width: 220 }}
              />
              <button
                className="btn-ghost"
                onClick={() => void pickWorkspace(setWorkspace)}
                title="Browse"
              >
                📂
              </button>
            </div>
            {sessionId && (
              <span className="session-badge" title={sessionId}>
                Session: {sessionId.slice(0, 8)}…
              </span>
            )}
            <button className="btn-ghost" onClick={newCeSession}>
              + New Session
            </button>
          </div>
        </div>

        <div className="ce-message-list">
          {chatMessages.length === 0 && (
            <div className="empty-state-sm">Chat with the Code Engine…</div>
          )}
          {chatMessages.map((m) => (
            <div key={m.id} className={`ce-msg ce-msg-${m.role} ${m.error ? "ce-msg-error" : ""}`}>
              <span className="ce-msg-role">{m.role === "user" ? "You" : "AI"}</span>
              <pre className="ce-msg-content">{m.content || (ceChatStreaming ? "▌" : "")}</pre>
            </div>
          ))}
          <div ref={chatBottomRef} />
        </div>

        <div className="ce-input-row">
          <textarea
            rows={2}
            className="ce-textarea"
            placeholder="Message the Code Engine… (Enter to send)"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={handleCeChatKey}
            disabled={ceChatStreaming}
          />
          {ceChatStreaming ? (
            <button className="btn-stop-sm" onClick={() => { ceChatEsRef.current?.close(); setCeChatStreaming(false); }}>
              ⏹
            </button>
          ) : (
            <button
              className="btn-send-sm"
              onClick={() => void sendCeChat()}
              disabled={!chatInput.trim()}
            >
              ▶
            </button>
          )}
        </div>
      </section>

      <div className="divider" />

      {/* ── Bottom: Xencode Compile ── */}
      <section className="xc-section">
        <div className="section-header">
          <span className="section-title">🚀 Xencode Compile</span>
          <div className="path-pick">
            <label>Workspace</label>
            <input
              type="text"
              placeholder="/output/workspace"
              value={xcWorkspace}
              onChange={(e) => setXcWorkspace(e.target.value)}
              style={{ width: 240 }}
            />
            <button
              className="btn-ghost"
              onClick={() => void pickWorkspace(setXcWorkspace)}
            >
              📂
            </button>
          </div>
        </div>

        <div className="xc-body">
          <div className="xc-spec-col">
            <label style={{ display: "block", marginBottom: 4 }}>
              Spec / README (paste your project description)
            </label>
            <textarea
              className="xc-spec-textarea"
              placeholder="Describe the project you want to build…"
              value={xcSpec}
              onChange={(e) => setXcSpec(e.target.value)}
              disabled={xcStreaming}
            />
            <div style={{ marginTop: 8 }}>
              {xcStreaming ? (
                <button
                  className="btn-stop"
                  onClick={() => { xcEsRef.current?.close(); setXcStreaming(false); }}
                >
                  ⏹ Stop
                </button>
              ) : (
                <button
                  className="btn-compile"
                  onClick={() => void runXencode()}
                  disabled={!xcSpec.trim()}
                >
                  ⚡ Compile Project
                </button>
              )}
            </div>

            {/* Output panel */}
            {xcResult && (
              <div className="output-panel">
                <div className="output-title">✅ Build Complete</div>
                {xcResult.zip_path && (
                  <div className="output-row">
                    <span className="output-label">ZIP</span>
                    <code className="output-path">{xcResult.zip_path}</code>
                    <button
                      className="btn-ghost"
                      onClick={() => void openInExplorer(xcResult.zip_path!)}
                    >
                      📂 Open
                    </button>
                  </div>
                )}
                {xcResult.exe_path && (
                  <div className="output-row">
                    <span className="output-label">EXE</span>
                    <code className="output-path">{xcResult.exe_path}</code>
                    <button
                      className="btn-ghost"
                      onClick={() => void openInExplorer(xcResult.exe_path!)}
                    >
                      📂 Open
                    </button>
                  </div>
                )}

                {/* GitHub Publish */}
                <div className="publish-section">
                  <div className="output-title" style={{ marginTop: 12 }}>
                    🐙 Publish to GitHub
                  </div>
                  <div className="publish-row">
                    <input
                      type="text"
                      placeholder="repo-name"
                      value={publishRepoName}
                      onChange={(e) => setPublishRepoName(e.target.value)}
                      style={{ width: 160 }}
                    />
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={publishPrivate}
                        onChange={(e) => setPublishPrivate(e.target.checked)}
                      />
                      Private
                    </label>
                    <button
                      className="btn-publish"
                      onClick={() => void publishToGitHub()}
                      disabled={publishing || !publishRepoName.trim()}
                    >
                      {publishing ? "Publishing…" : "Publish"}
                    </button>
                  </div>
                  {publishResult && (
                    <div className="publish-result">{publishResult}</div>
                  )}
                </div>
              </div>
            )}

            {xcError && <div className="xc-error">❌ {xcError}</div>}
          </div>

          <div className="xc-log-col">
            <StreamLog entries={xcLogs} streaming={xcStreaming} />
          </div>
        </div>
      </section>

      <style>{`
        .coding-mode {
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
        }
        .offline-banner {
          background: #3a1a1a;
          color: #f87171;
          padding: 6px 14px;
          font-size: 0.8rem;
          border-bottom: 1px solid #5a2020;
          flex-shrink: 0;
        }
        .offline-banner code { color: #fbbf24; }

        /* Sections */
        .ce-section {
          display: flex;
          flex-direction: column;
          flex: 0 0 45%;
          overflow: hidden;
          border-bottom: 1px solid #222;
        }
        .xc-section {
          display: flex;
          flex-direction: column;
          flex: 1;
          overflow: hidden;
          min-height: 0;
        }
        .divider { height: 4px; background: #1a1a2e; flex-shrink: 0; }

        .section-header {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 14px;
          background: #14142a;
          border-bottom: 1px solid #222;
          flex-shrink: 0;
          flex-wrap: wrap;
        }
        .section-title {
          font-weight: 600;
          color: #c4b8ff;
          font-size: 0.88rem;
          margin-right: 4px;
        }
        .ce-header-controls {
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
        }
        .ce-header-controls label { font-size: 0.78rem; color: #888; }
        .ce-header-controls select {
          padding: 3px 6px;
          font-size: 0.8rem;
          background: #1e1e30;
          border: 1px solid #333;
          color: #ddd;
          border-radius: 5px;
        }
        .path-pick { display: flex; align-items: center; gap: 4px; }
        .session-badge {
          background: #1a2040;
          border: 1px solid #3040a0;
          color: #8898d8;
          padding: 2px 8px;
          border-radius: 20px;
          font-size: 0.75rem;
          font-family: monospace;
        }

        /* CE Chat */
        .ce-message-list {
          flex: 1;
          overflow-y: auto;
          padding: 10px 14px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .empty-state-sm { color: #444; font-size: 0.82rem; margin: auto; }
        .ce-msg {
          display: flex;
          gap: 8px;
          align-items: flex-start;
        }
        .ce-msg-user { flex-direction: row-reverse; }
        .ce-msg-role {
          font-size: 0.72rem;
          font-weight: 700;
          color: #7c6af7;
          flex-shrink: 0;
          padding-top: 3px;
          width: 28px;
          text-align: center;
        }
        .ce-msg-user .ce-msg-role { color: #4a9eff; }
        .ce-msg-content {
          background: #1a1a2e;
          border: 1px solid #2d2d50;
          border-radius: 8px;
          padding: 7px 11px;
          white-space: pre-wrap;
          word-break: break-word;
          font-family: inherit;
          font-size: 0.85rem;
          line-height: 1.5;
          color: #ddd;
          margin: 0;
          max-width: 600px;
        }
        .ce-msg-user .ce-msg-content { background: #1a2240; border-color: #2a3a80; }
        .ce-msg-error .ce-msg-content { border-color: #7f2020; background: #2a1010; color: #f87171; }

        .ce-input-row {
          display: flex;
          gap: 6px;
          padding: 8px 14px;
          background: #12121f;
          flex-shrink: 0;
          align-items: flex-end;
        }
        .ce-textarea {
          flex: 1;
          resize: none;
          background: #1a1a2e;
          border: 1px solid #333;
          color: #e0e0e0;
          padding: 7px 10px;
          border-radius: 6px;
          font-size: 0.85rem;
          line-height: 1.5;
        }
        .btn-send-sm, .btn-stop-sm {
          padding: 7px 14px;
          border-radius: 6px;
          border: none;
          cursor: pointer;
          font-size: 0.9rem;
          height: fit-content;
        }
        .btn-send-sm { background: #7c6af7; color: #fff; }
        .btn-send-sm:hover:not(:disabled) { background: #6a58e0; }
        .btn-send-sm:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-stop-sm { background: #8b2020; color: #fff; }

        /* Xencode */
        .xc-body {
          display: flex;
          flex: 1;
          overflow: hidden;
          gap: 0;
        }
        .xc-spec-col {
          flex: 0 0 380px;
          display: flex;
          flex-direction: column;
          padding: 12px 14px;
          overflow-y: auto;
          border-right: 1px solid #222;
        }
        .xc-log-col {
          flex: 1;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }
        .xc-spec-textarea {
          flex: 1;
          min-height: 100px;
          max-height: 180px;
          resize: vertical;
          background: #1a1a2e;
          border: 1px solid #333;
          color: #e0e0e0;
          padding: 8px 10px;
          border-radius: 6px;
          font-size: 0.85rem;
          line-height: 1.5;
        }
        .btn-compile {
          background: #1a6040;
          color: #7effc7;
          border: 1px solid #2a8050;
          padding: 7px 18px;
          border-radius: 7px;
          cursor: pointer;
          font-size: 0.88rem;
          font-weight: 600;
          transition: background 0.15s;
        }
        .btn-compile:hover:not(:disabled) { background: #1e7050; }
        .btn-compile:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-stop {
          background: #8b2020; color: #fff; border: none;
          padding: 7px 18px; border-radius: 7px; cursor: pointer;
        }

        .output-panel {
          margin-top: 14px;
          background: #0f1a14;
          border: 1px solid #2a4a30;
          border-radius: 8px;
          padding: 12px 14px;
        }
        .output-title { font-size: 0.82rem; font-weight: 700; color: #7effc7; margin-bottom: 8px; }
        .output-row { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
        .output-label {
          font-size: 0.72rem; font-weight: 700; color: #aaa;
          background: #1a2a1a; border-radius: 4px; padding: 1px 6px;
          border: 1px solid #2a4a2a;
        }
        .output-path {
          font-size: 0.78rem; color: #88ffcc;
          overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
          max-width: 180px;
        }

        .publish-section { margin-top: 8px; }
        .publish-row { display: flex; align-items: center; gap: 8px; margin-top: 6px; flex-wrap: wrap; }
        .checkbox-label {
          display: flex; align-items: center; gap: 5px;
          font-size: 0.8rem; color: #aaa; cursor: pointer;
        }
        .btn-publish {
          background: #24292f; color: #fff; border: 1px solid #444;
          padding: 5px 14px; border-radius: 6px; cursor: pointer; font-size: 0.82rem;
        }
        .btn-publish:hover:not(:disabled) { background: #30363d; }
        .btn-publish:disabled { opacity: 0.5; cursor: not-allowed; }
        .publish-result { font-size: 0.8rem; color: #7effc7; margin-top: 6px; word-break: break-all; }

        .xc-error {
          margin-top: 10px; color: #f87171; font-size: 0.82rem;
          background: #2a1010; border: 1px solid #7f2020;
          border-radius: 6px; padding: 8px 10px;
        }

        .btn-ghost {
          background: transparent; border: 1px solid #333; color: #888;
          padding: 3px 9px; border-radius: 5px; cursor: pointer; font-size: 0.78rem;
        }
        .btn-ghost:hover { border-color: #555; color: #bbb; }
      `}</style>
    </div>
  );
}
