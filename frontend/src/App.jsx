import { useMemo, useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

const starterMessages = [];


function nowTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function parseSystemHint(raw) {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    const systemHint = typeof parsed.system_hint === "string" ? parsed.system_hint.trim() : "";
    if (!systemHint) return null;
    const label = typeof parsed.label === "string" ? parsed.label.trim() : "";
    const suggestedExamples = Array.isArray(parsed.suggested_examples)
      ? parsed.suggested_examples
          .filter((example) => typeof example === "string" && example.trim())
          .map((example) => example.trim())
      : [];
    return { label, systemHint, suggestedExamples };
  } catch {
    return null;
  }
}

function App() {
  const [messages, setMessages] = useState(starterMessages);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);

  const placeholderTips = useMemo(
    () => ["過去五天成交量下降", "產生一個alpha並將rmse當作fitness function",],
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
      const notice = (data.notice || "").trim();
      const systemHint = (data.system_hint || "").trim();
      const parsedSystemHint = parseSystemHint(systemHint);
      const report = (data.evaluation_report || "").trim();
      if (!notice && !systemHint && !report) {
        throw new Error("Backend returned empty evaluation report");
      }

      const assistantMessages = [];
      if (notice) {
        assistantMessages.push({
          id: Date.now() + 1,
          role: "assistant",
          content: notice,
          time: nowTime(),
        });
      }
      if (parsedSystemHint) {
        assistantMessages.push({
          id: Date.now() + 2,
          role: "assistant",
          content: "",
          type: "system_hint",
          payload: parsedSystemHint,
          time: nowTime(),
        });
      } else if (systemHint && systemHint !== notice) {
        assistantMessages.push({
          id: Date.now() + 2,
          role: "assistant",
          content: systemHint,
          time: nowTime(),
        });
      }
      if (report && report !== systemHint) {
        assistantMessages.push({
          id: Date.now() + 3,
          role: "assistant",
          content: report,
          time: nowTime(),
        });
      }

      setMessages((prev) => [
        ...prev,
        ...assistantMessages,
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
        <h1>Alpha Mining</h1>
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
                {message.role === "assistant" ? (
                  message.type === "system_hint" ? (
                    <div className="system-hint-card">
                      <div className="system-hint-header">
                        <span className="system-hint-title">系統操作提示</span>
                        {message.payload?.label ? (
                          <span className="system-hint-label">{message.payload.label}</span>
                        ) : null}
                      </div>
                      <p className="system-hint-text">{message.payload?.systemHint}</p>
                      {message.payload?.suggestedExamples?.length ? (
                        <div className="system-hint-examples">
                          <div className="system-hint-subtitle">建議範例</div>
                          <ul>
                            {message.payload.suggestedExamples.map((example, idx) => (
                              <li key={`${message.id}-ex-${idx}`}>{example}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div className="markdown-content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                    </div>
                  )
                ) : (
                  <p>{message.content}</p>
                )}
                <span>{message.time}</span>
              </div>
            </article>
          ))}
        </section>

        <footer className="composer-wrap">
          {/* <div className="tips" aria-hidden="true">
            {placeholderTips.map((tip) => (
              <button key={tip} type="button" onClick={() => setDraft(tip)}>
                {tip}
              </button>
            ))}
          </div> */}

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
