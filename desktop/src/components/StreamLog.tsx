import { useEffect, useRef } from "react";

export interface LogEntry {
  type: string;
  message: string;
  timestamp?: string;
}

interface StreamLogProps {
  entries: LogEntry[];
  streaming?: boolean;
  maxHeight?: string;
}

const EVENT_COLORS: Record<string, string> = {
  spec_parsed: "#60a5fa",       // blue
  plan: "#22d3ee",              // cyan
  files_written: "#4ade80",     // green
  file_written: "#4ade80",
  validation: "#facc15",        // yellow
  validate: "#facc15",
  final_build_started: "#fb923c", // orange
  build: "#fb923c",
  final_build_output: "#e5e7eb", // white-ish
  done: "#86efac",              // bright green
  complete: "#86efac",
  error: "#f87171",             // red
  info: "#9ca3af",              // gray
  warning: "#fbbf24",           // amber
  debug: "#6b7280",             // dark gray
};

function getColor(type: string): string {
  const key = type.toLowerCase();
  for (const [k, v] of Object.entries(EVENT_COLORS)) {
    if (key.includes(k)) return v;
  }
  return EVENT_COLORS.info;
}

function getPrefix(type: string): string {
  const key = type.toLowerCase();
  if (key.includes("error")) return "✗";
  if (key.includes("done") || key.includes("complete")) return "✓";
  if (key.includes("warn")) return "⚠";
  if (key.includes("plan")) return "◆";
  if (key.includes("spec")) return "◈";
  if (key.includes("file")) return "◉";
  if (key.includes("build")) return "⚡";
  if (key.includes("valid")) return "◎";
  return "›";
}

export default function StreamLog({ entries, streaming = false, maxHeight = "100%" }: StreamLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  return (
    <div className="stream-log" style={{ maxHeight }}>
      <div className="sl-header">
        <span className="sl-title">📡 Live Log</span>
        {streaming && (
          <span className="sl-pulse">
            <span className="pulse-dot" /> streaming
          </span>
        )}
        <span className="sl-count">{entries.length} events</span>
      </div>

      <div className="sl-body">
        {entries.length === 0 && !streaming && (
          <div className="sl-empty">
            Waiting for events… Run a compile to see output here.
          </div>
        )}
        {entries.map((entry, i) => (
          <div key={i} className="sl-entry">
            <span className="sl-type" style={{ color: getColor(entry.type) }}>
              {getPrefix(entry.type)} [{entry.type}]
            </span>
            {entry.timestamp && (
              <span className="sl-time">{entry.timestamp}</span>
            )}
            <span className="sl-msg" style={{ color: getColor(entry.type) }}>
              {entry.message}
            </span>
          </div>
        ))}
        {streaming && (
          <div className="sl-entry sl-entry-cursor">
            <span style={{ color: "#9ca3af" }}>› </span>
            <span className="cursor-blink">▌</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <style>{`
        .stream-log {
          display: flex;
          flex-direction: column;
          height: 100%;
          background: #080d10;
          font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
          overflow: hidden;
        }
        .sl-header {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 6px 12px;
          border-bottom: 1px solid #1a2a1a;
          background: #0a1210;
          flex-shrink: 0;
        }
        .sl-title { font-size: 0.8rem; font-weight: 600; color: #4ade80; }
        .sl-pulse {
          display: flex;
          align-items: center;
          gap: 5px;
          font-size: 0.72rem;
          color: #4ade80;
        }
        .pulse-dot {
          width: 7px;
          height: 7px;
          background: #4ade80;
          border-radius: 50%;
          animation: pulse 1s infinite;
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(0.8); }
        }
        .sl-count { margin-left: auto; font-size: 0.72rem; color: #374151; }

        .sl-body {
          flex: 1;
          overflow-y: auto;
          padding: 8px 12px;
          display: flex;
          flex-direction: column;
          gap: 3px;
        }
        .sl-empty { color: #374151; font-size: 0.8rem; margin: auto; }
        .sl-entry {
          display: flex;
          align-items: baseline;
          gap: 8px;
          line-height: 1.6;
          font-size: 0.8rem;
        }
        .sl-type {
          flex-shrink: 0;
          font-weight: 600;
          font-size: 0.75rem;
          min-width: 140px;
        }
        .sl-time {
          flex-shrink: 0;
          color: #374151;
          font-size: 0.7rem;
        }
        .sl-msg {
          flex: 1;
          word-break: break-word;
          white-space: pre-wrap;
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        .cursor-blink { animation: blink 1s step-end infinite; }
      `}</style>
    </div>
  );
}
