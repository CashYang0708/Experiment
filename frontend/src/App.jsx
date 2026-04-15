import { useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

const starterMessages = [];

function nowTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function App() {
  const [messages, setMessages] = useState(starterMessages);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);

  const placeholderTips = useMemo(
    () => ["Summarize this file...", "Generate a test plan...", "Refactor this component..."],
    []
  );

  const sendMessage = async () => {
    const trimmed = draft.trim();
    if (!trimmed || isSending) return;

    const userMessage = {
      id: Date.now(),
      role: "user",
      content: trimmed,
      time: nowTime(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setDraft("");

    try {
      setIsSending(true);
      const response = await fetch(`${API_BASE_URL}/evaluate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: trimmed }),
      });

      if (!response.ok) {
        let detail = "Request failed";
        try {
          const errorPayload = await response.json();
          detail = errorPayload.detail || detail;
        } catch {
          // Ignore parse errors and keep fallback text.
        }
        throw new Error(detail);
      }

      const data = await response.json();
      const report = (data.evaluation_report || "").trim();
      if (!report) {
        throw new Error("Backend returned empty evaluation report");
      }

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: report,
          time: nowTime(),
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: `Failed to fetch backend response: ${error.message}`,
          time: nowTime(),
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const onKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="page-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <header className="top-bar">
        <h1>Companion Chat</h1>
        <button type="button" className="new-chat-btn" onClick={() => setMessages(starterMessages)}>
          New Chat
        </button>
      </header>

      <main className="chat-frame">
        <section className="messages" aria-label="Chat messages">
          {messages.map((message, index) => (
            <article
              key={message.id}
              className={`bubble-row ${message.role}`}
              style={{ animationDelay: `${index * 80}ms` }}
            >
              <div className="avatar">{message.role === "assistant" ? "AI" : "ME"}</div>
              <div className="bubble-content">
                <p>{message.content}</p>
                <span>{message.time}</span>
              </div>
            </article>
          ))}
        </section>

        <footer className="composer-wrap">
          <div className="tips" aria-hidden="true">
            {placeholderTips.map((tip) => (
              <button key={tip} type="button" onClick={() => setDraft(tip)}>
                {tip}
              </button>
            ))}
          </div>

          <div className="composer">
            <textarea
              rows={1}
              value={draft}
              disabled={isSending}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Message Companion Chat"
              aria-label="Type your message"
            />
            <button type="button" onClick={sendMessage} disabled={!draft.trim() || isSending}>
              {isSending ? "Sending..." : "Send"}
            </button>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
