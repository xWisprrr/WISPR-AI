import { useState } from "react";
import ChatMode from "./components/ChatMode";
import CodingMode from "./components/CodingMode";
import "./App.css";

type Tab = "chat" | "coding";

export const BACKEND_URL =
  (import.meta.env.VITE_BACKEND_URL as string | undefined) ??
  "http://localhost:8000";

function App() {
  const [activeTab, setActiveTab] = useState<Tab>("chat");

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-logo">
          <span className="logo-text">WISPR</span>
          <span className="logo-sub">Desktop Studio</span>
        </div>
        <nav className="tab-bar">
          <button
            className={`tab-btn ${activeTab === "chat" ? "active" : ""}`}
            onClick={() => setActiveTab("chat")}
          >
            💬 Chat
          </button>
          <button
            className={`tab-btn ${activeTab === "coding" ? "active" : ""}`}
            onClick={() => setActiveTab("coding")}
          >
            ⚙️ Coding
          </button>
        </nav>
        <div className="backend-indicator">
          <span className="backend-url">{BACKEND_URL}</span>
        </div>
      </header>

      <main className="app-main">
        {activeTab === "chat" && <ChatMode />}
        {activeTab === "coding" && <CodingMode />}
      </main>
    </div>
  );
}

export default App;
