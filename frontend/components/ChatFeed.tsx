import { useState, useRef, useEffect } from "react";
import QValuesGraph from "./QValuesGraph";

export type HeadType = "FACTUAL" | "PROCEDURAL" | "CONCEPTUAL" | "TRANSFER";
export type ActionType = "explain" | "correct_fact" | "worked_example" | "analogize" | "question";

export interface ChatMessage {
  id: string;
  role: "tutor" | "student";
  content: string;
  head?: HeadType;
  action?: ActionType;
  timestamp: number;
  reward?: number;
  handlingScore?: number;
  qValues?: Record<string, number>;
}

const HEAD_COLORS: Record<HeadType, { bg: string; border: string; badge: string; text: string }> = {
  FACTUAL:     { bg: "#0d1f2d", border: "#1e6fa5", badge: "#1e6fa5", text: "#5bc4f5" },
  PROCEDURAL:  { bg: "#1a1a0d", border: "#8a7a10", badge: "#c4a81e", text: "#f5e45b" },
  CONCEPTUAL:  { bg: "#1a0d1a", border: "#7a1e8a", badge: "#b42ecf", text: "#e87af5" },
  TRANSFER:    { bg: "#0d1a0d", border: "#1e8a3a", badge: "#2acf5a", text: "#5bf587" },
};

const ACTION_ICONS: Record<ActionType, string> = {
  explain:       "📖",
  correct_fact:  "✅",
  worked_example:"🔧",
  analogize:     "🔗",
  question:      "❓",
};

const ACTION_LABELS: Record<ActionType, string> = {
  explain:       "Explain",
  correct_fact:  "Correct",
  worked_example:"Example",
  analogize:     "Analogy",
  question:      "Query",
};

interface ChatFeedProps {
  messages: ChatMessage[];
  isThinking?: boolean;
}

export default function ChatFeed({ messages, isThinking = false }: ChatFeedProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "16px",
        padding: "20px 16px",
        height: "100%",
        overflowY: "auto",
        background: "#080c10",
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        scrollbarWidth: "thin",
        scrollbarColor: "#1e3a50 transparent",
      }}
    >
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      {isThinking && <ThinkingIndicator />}
      <div ref={bottomRef} />
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isTutor = message.role === "tutor";

  if (!isTutor) {
    return (
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <div
          style={{
            maxWidth: "72%",
            padding: "10px 14px",
            background: "#111820",
            border: "1px solid #243040",
            borderRadius: "12px 2px 12px 12px",
            color: "#9ab8cc",
            fontSize: "13px",
            lineHeight: "1.6",
          }}
        >
          {message.content}
          <div style={{ fontSize: "10px", color: "#3a5060", marginTop: "6px", textAlign: "right" }}>
            {new Date(message.timestamp).toLocaleTimeString()}
          </div>
        </div>
      </div>
    );
  }

  const headKey = message.head ?? "FACTUAL";
  const colors = HEAD_COLORS[headKey];

  return (
    <div style={{ display: "flex", justifyContent: "flex-start" }}>
      <div
        style={{
          maxWidth: "82%",
          background: colors.bg,
          border: `1px solid ${colors.border}`,
          borderRadius: "2px 12px 12px 12px",
          overflow: "hidden",
          boxShadow: `0 0 16px ${colors.border}22`,
        }}
      >
        {/* Header row */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            padding: "7px 12px",
            borderBottom: `1px solid ${colors.border}55`,
            background: `${colors.border}18`,
          }}
        >
          {/* Head tag */}
          <span
            style={{
              fontSize: "9px",
              fontWeight: 700,
              letterSpacing: "0.12em",
              color: colors.text,
              background: `${colors.badge}22`,
              border: `1px solid ${colors.badge}66`,
              padding: "2px 7px",
              borderRadius: "3px",
            }}
          >
            [{headKey}_HEAD]
          </span>

          {/* Action badge */}
          {message.action && (
            <span
              style={{
                display: "flex",
                alignItems: "center",
                gap: "4px",
                fontSize: "10px",
                color: "#7a9aaa",
                background: "#ffffff08",
                border: "1px solid #2a3a44",
                padding: "2px 8px",
                borderRadius: "3px",
              }}
            >
              <span>{ACTION_ICONS[message.action]}</span>
              <span>{ACTION_LABELS[message.action]}</span>
            </span>
          )}

          {/* Reward pill */}
          {message.reward !== undefined && (
            <span
              style={{
                marginLeft: "auto",
                fontSize: "10px",
                fontWeight: 700,
                color: message.reward >= 0 ? "#2acf5a" : "#cf4a2a",
                background: message.reward >= 0 ? "#0a2a1440" : "#2a0a0440",
                border: `1px solid ${message.reward >= 0 ? "#2acf5a44" : "#cf4a2a44"}`,
                padding: "2px 8px",
                borderRadius: "3px",
              }}
            >
              {message.reward >= 0 ? "+" : ""}{message.reward.toFixed(1)} r
            </span>
          )}

          {/* Handling Score pill */}
          {message.handlingScore !== undefined && (
            <span
              style={{
                marginLeft: message.reward !== undefined ? "8px" : "auto",
                fontSize: "10px",
                fontWeight: 700,
                color: "#5bc4f5",
                background: "#0a1a2a40",
                border: "1px solid #1e6fa544",
                padding: "2px 8px",
                borderRadius: "3px",
              }}
            >
              Handled: {message.handlingScore}%
            </span>
          )}
        </div>

        {/* Body */}
        <div
          style={{
            padding: "12px 14px",
            color: "#c8dce8",
            fontSize: "13px",
            lineHeight: "1.7",
          }}
        >
          {message.content}
          
          {/* Dynamic Response Graph */}
          {message.role === "tutor" && message.qValues && (
            <QValuesGraph 
              qValues={message.qValues} 
              selectedAction={message.action} 
              handlingScore={message.handlingScore}
            />
          )}
        </div>

        <div
          style={{
            padding: "4px 14px 8px",
            fontSize: "10px",
            color: "#2a4050",
          }}
        >
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <div style={{ display: "flex", justifyContent: "flex-start" }}>
      <div
        style={{
          padding: "10px 16px",
          background: "#0d1820",
          border: "1px solid #1e3a50",
          borderRadius: "2px 12px 12px 12px",
          display: "flex",
          alignItems: "center",
          gap: "6px",
        }}
      >
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            style={{
              width: "6px",
              height: "6px",
              borderRadius: "50%",
              background: "#1e6fa5",
              display: "inline-block",
              animation: "pulse 1.2s ease-in-out infinite",
              animationDelay: `${i * 0.2}s`,
            }}
          />
        ))}
        <style>{`@keyframes pulse { 0%,80%,100%{opacity:.2;transform:scale(.8)} 40%{opacity:1;transform:scale(1)} }`}</style>
      </div>
    </div>
  );
}
