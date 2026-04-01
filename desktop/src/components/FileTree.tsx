import { useState, useEffect, useCallback } from "react";
import { BACKEND_URL } from "../App";

interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
  children?: FileEntry[];
}

interface FileTreeProps {
  onFileSelect?: (path: string, content: string) => void;
  rootPath?: string;
}

export default function FileTree({ onFileSelect, rootPath = "" }: FileTreeProps) {
  const [tree, setTree] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selectedPath, setSelectedPath] = useState<string>("");

  const loadTree = useCallback(
    async (path: string = rootPath) => {
      setLoading(true);
      setError("");
      try {
        const params = new URLSearchParams(path ? { path } : {});
        const res = await fetch(`${BACKEND_URL}/code-engine/files/list?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as { files?: FileEntry[]; tree?: FileEntry[] } | FileEntry[];
        const entries = Array.isArray(data)
          ? data
          : (data as { files?: FileEntry[]; tree?: FileEntry[] }).files ??
            (data as { tree?: FileEntry[] }).tree ??
            [];
        setTree(entries);
      } catch (err) {
        setError(`Failed to load files: ${String(err)}`);
      } finally {
        setLoading(false);
      }
    },
    [rootPath]
  );

  useEffect(() => {
    void loadTree();
  }, [loadTree]);

  const readFile = async (path: string) => {
    try {
      const params = new URLSearchParams({ path });
      const res = await fetch(`${BACKEND_URL}/code-engine/files/read?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as { content?: string };
      onFileSelect?.(path, data.content ?? "");
    } catch (err) {
      setError(`Failed to read file: ${String(err)}`);
    }
  };

  const toggleDir = (path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(path) ? next.delete(path) : next.add(path);
      return next;
    });
  };

  const handleEntry = (entry: FileEntry) => {
    if (entry.is_dir) {
      toggleDir(entry.path);
    } else {
      setSelectedPath(entry.path);
      void readFile(entry.path);
    }
  };

  const renderEntries = (entries: FileEntry[], depth = 0): React.ReactNode =>
    entries.map((entry) => {
      const isOpen = expanded.has(entry.path);
      const isSelected = selectedPath === entry.path;
      return (
        <div key={entry.path}>
          <div
            className={`ft-entry ${isSelected ? "ft-selected" : ""}`}
            style={{ paddingLeft: `${depth * 14 + 8}px` }}
            onClick={() => handleEntry(entry)}
            title={entry.path}
          >
            <span className="ft-icon">
              {entry.is_dir ? (isOpen ? "📂" : "📁") : getFileIcon(entry.name)}
            </span>
            <span className="ft-name">{entry.name}</span>
          </div>
          {entry.is_dir && isOpen && entry.children && (
            <div>{renderEntries(entry.children, depth + 1)}</div>
          )}
        </div>
      );
    });

  return (
    <div className="file-tree">
      <div className="ft-toolbar">
        <span className="ft-title">📁 Files</span>
        <button className="ft-refresh" onClick={() => void loadTree()} title="Refresh">
          🔄
        </button>
      </div>

      {loading && <div className="ft-status">Loading…</div>}
      {error && <div className="ft-error">{error}</div>}
      {!loading && tree.length === 0 && !error && (
        <div className="ft-status">No files found.</div>
      )}
      <div className="ft-list">{renderEntries(tree)}</div>

      <style>{`
        .file-tree {
          display: flex;
          flex-direction: column;
          height: 100%;
          background: #111120;
          border-right: 1px solid #222;
          min-width: 180px;
          max-width: 280px;
          overflow: hidden;
        }
        .ft-toolbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 7px 10px;
          border-bottom: 1px solid #222;
          flex-shrink: 0;
        }
        .ft-title { font-size: 0.8rem; font-weight: 600; color: #aaa; }
        .ft-refresh {
          background: transparent; border: none; cursor: pointer;
          font-size: 0.9rem; color: #666; padding: 0;
        }
        .ft-refresh:hover { color: #aaa; }
        .ft-list { flex: 1; overflow-y: auto; }
        .ft-status { padding: 10px; color: #555; font-size: 0.8rem; }
        .ft-error { padding: 8px; color: #f87171; font-size: 0.78rem; }
        .ft-entry {
          display: flex;
          align-items: center;
          gap: 5px;
          padding-top: 4px;
          padding-bottom: 4px;
          padding-right: 8px;
          cursor: pointer;
          white-space: nowrap;
          overflow: hidden;
          border-radius: 4px;
          margin: 1px 4px 1px 0;
        }
        .ft-entry:hover { background: #1e1e30; }
        .ft-selected { background: #1a2050 !important; }
        .ft-icon { font-size: 0.85rem; flex-shrink: 0; }
        .ft-name {
          font-size: 0.82rem;
          color: #ccc;
          overflow: hidden;
          text-overflow: ellipsis;
        }
      `}</style>
    </div>
  );
}

function getFileIcon(name: string): string {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = {
    ts: "🔷", tsx: "⚛️", js: "🟨", jsx: "⚛️",
    py: "🐍", rs: "🦀", go: "🐹", java: "☕",
    json: "📋", md: "📝", toml: "⚙️", yaml: "⚙️", yml: "⚙️",
    html: "🌐", css: "🎨", sh: "🖥️", txt: "📄",
    png: "🖼️", jpg: "🖼️", jpeg: "🖼️", svg: "🖼️",
    zip: "📦", exe: "⚙️",
  };
  return map[ext] ?? "📄";
}
