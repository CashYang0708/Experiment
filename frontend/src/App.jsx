import { useMemo, useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

const starterMessages = [];

const HARDCODED_COMMANDS = [
  { id: "restart_worker", description: "本系統為alpha mining的系統架構，會有兩個agent處理不同的任務，alpha search agent以及GP agent，分別負責從現有的alpha資料庫中找尋符合市場情境的alpha以及根據你的交易想法生成新的alpha，找到或生成的alpha並經過回測驗證確認其有效性。當你對系統提出一個查詢時，系統會先分析你的查詢內容，判斷你是想要找尋現有的alpha還是想要生成新的alpha，然後將任務分配給相對應的agent來處理。", 
    command: "Alpha_search Agent指令:過去五天成交量下降\nGP Agent指令:產生一個均值回歸的alpha並將rmse當作fitness function" },
];

function nowTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
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

  const opsInserted = useRef(false);
  useEffect(() => {
    // Guard to avoid double insertion in React.StrictMode (dev)
    if (opsInserted.current) return;
    opsInserted.current = true;

    if (!HARDCODED_COMMANDS || HARDCODED_COMMANDS.length === 0) return;
    const opsMessages = HARDCODED_COMMANDS.map((c, idx) => {
      const cmdLines = (c.command || "").split("\n").map((l) => `**${l}**`).join("\n\n");
      return {
        id: `sys-${c.id}-${Date.now()}-${idx}`,
        role: "assistant",
        content: `${c.description} 以下為你可以參考的指令:\n\n${cmdLines}`,
        time: nowTime(),
      };
    });
    setMessages((prev) => [...opsMessages, ...prev]);
  }, []);

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
                  <div className="markdown-content">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                  </div>
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
