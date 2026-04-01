import { useRef, useEffect, lazy, Suspense, useState } from "react";

// Try to dynamically import @monaco-editor/react. If it's not installed,
// we fall back to a plain textarea editor.
const MonacoReact = lazy(() =>
  import("@monaco-editor/react").catch(() => ({
    default: () => null,
  }))
);

export interface MonacoEditorProps {
  value: string;
  onChange?: (value: string) => void;
  language?: string;
  readOnly?: boolean;
  height?: string;
}

function TextareaFallback({ value, onChange, readOnly, height = "300px" }: MonacoEditorProps) {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange?.(e.target.value)}
      readOnly={readOnly}
      style={{
        width: "100%",
        height,
        background: "#1e1e2e",
        color: "#d4d4d4",
        border: "1px solid #333",
        borderRadius: "6px",
        padding: "10px 12px",
        fontFamily: '"JetBrains Mono", "Fira Code", Consolas, monospace',
        fontSize: "0.85rem",
        lineHeight: 1.6,
        resize: "vertical",
        outline: "none",
        boxSizing: "border-box",
      }}
    />
  );
}

let monacoAvailable: boolean | null = null;

export default function MonacoEditor(props: MonacoEditorProps) {
  const { value, onChange, language = "plaintext", readOnly = false, height = "300px" } = props;
  const [useMonaco, setUseMonaco] = useState<boolean | null>(null);
  const checkedRef = useRef(false);

  useEffect(() => {
    if (checkedRef.current) return;
    checkedRef.current = true;

    if (monacoAvailable !== null) {
      setUseMonaco(monacoAvailable);
      return;
    }

    // Probe whether @monaco-editor/react is importable
    import("@monaco-editor/react")
      .then(() => {
        monacoAvailable = true;
        setUseMonaco(true);
      })
      .catch(() => {
        monacoAvailable = false;
        setUseMonaco(false);
      });
  }, []);

  if (useMonaco === null) {
    return (
      <div
        style={{
          height,
          background: "#1e1e2e",
          border: "1px solid #333",
          borderRadius: "6px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#555",
          fontSize: "0.8rem",
        }}
      >
        Loading editor…
      </div>
    );
  }

  if (!useMonaco) {
    return <TextareaFallback {...props} />;
  }

  return (
    <Suspense fallback={<TextareaFallback {...props} />}>
      <MonacoReact
        height={height}
        language={language}
        value={value}
        onChange={(val) => onChange?.(val ?? "")}
        theme="vs-dark"
        options={{
          readOnly,
          minimap: { enabled: false },
          fontSize: 13,
          wordWrap: "on",
          scrollBeyondLastLine: false,
          lineNumbersMinChars: 3,
          renderLineHighlight: "line",
          padding: { top: 8, bottom: 8 },
        }}
      />
    </Suspense>
  );
}
