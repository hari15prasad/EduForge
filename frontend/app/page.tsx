"use client";

import { useState, useCallback, useEffect, useRef } from "react";

import ChatFeed, {
  type ChatMessage,
  type HeadType,
  type ActionType,
} from "@/components/ChatFeed";

import HeadSwitcher, { type HeadStats } from "@/components/HeadSwitcher";

import SimulatorInput, {
  type StudentState,
  type WhatIfResult,
} from "@/components/SimulatorInput";
import ActionPanel from "@/components/ActionPanel";

import { dqnClient } from "@/lib/dqn-client";
import { useRL } from "@/components/RLProvider";

// ── Initial student state ────────────────────────────────────────────────────

const INITIAL_STATE: StudentState = {
  confusion:          7.5,
  attention:          6.0,
  domain_index:       0,
  step_count:         0,
  last_action_impact: 0,
};

const DOMAIN_HEAD: Record<number, HeadType> = {
  0: "FACTUAL",
  1: "PROCEDURAL",
  2: "CONCEPTUAL",
  3: "TRANSFER",
};

// ── Types ────────────────────────────────────────────────────────────────────

type SidebarTab = "state" | "sim";

// ── Page ─────────────────────────────────────────────────────────────────────

export default function EduForgePage() {
  const { applyStepUpdate, resetEpisode } = useRL();
  const [messages,      setMessages]      = useState<ChatMessage[]>([]);
  const [studentState,  setStudentState]  = useState<StudentState>(INITIAL_STATE);
  const [activeHead,    setActiveHead]    = useState<HeadType>("FACTUAL");
  const [trainingStats, setTrainingStats] = useState<HeadStats[]>([]);
  const [isThinking,    setIsThinking]    = useState(false);
  const [sidebarTab,    setSidebarTab]    = useState<SidebarTab>("state");
  const [episodeActive, setEpisodeActive] = useState(true);
  const [isAutoMode,    setIsAutoMode]    = useState(false);
  const [userInput,     setUserInput]     = useState("");
  const [sessionId,     setSessionId]     = useState<string>(() => crypto.randomUUID());
  const [lastQValues,   setLastQValues]   = useState<Record<string, number>>({});
  const stepTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Refresh stats every 5 s
  useEffect(() => {
    setTrainingStats(dqnClient.getTrainingStats());
    const id = setInterval(() => setTrainingStats(dqnClient.getTrainingStats()), 5000);
    return () => clearInterval(id);
  }, []);

  // Auto-step: tutor fires a move every 3 s when episode is active
  const fireStep = useCallback(async (state: StudentState, studentMsgText?: string, forcedAction?: ActionType) => {
    if (isThinking) return;
    setIsThinking(true);

    try {
      const result = await dqnClient.forward(state, studentMsgText, forcedAction, sessionId);

      const tutorMsg: ChatMessage = {
        id:            crypto.randomUUID(),
        role:          "tutor",
        content:       result.generated_text || dqnClient.getTutorMessage(result.action),
        head:          result.selected_head,
        action:        result.action as ActionType,
        reward:        result.expected_reward,
        handlingScore: result.handling_score,
        qValues:       result.q_values,
        timestamp:     Date.now(),
      };

      setMessages((prev) => [...prev, tutorMsg]);
      setStudentState(result.next_state);
      setActiveHead(result.selected_head);

      applyStepUpdate(
        {
          confusion: result.next_state.confusion,
          attention: result.next_state.attention,
          domainIndex: result.next_state.domain_index,
          currentStep: result.next_state.step_count,
          lastActionImpact: result.next_state.last_action_impact
        },
        {
          step: result.next_state.step_count,
          reward: result.expected_reward,
          action: result.action as any,
          timestamp: Date.now()
        }
      );

      if (result.terminated) {
        setEpisodeActive(false);
        const sys: ChatMessage = {
          id:        crypto.randomUUID(),
          role:      "tutor",
          content:   result.termination_reason === "success"
            ? "✓ Episode complete — confusion target reached. Starting new episode…"
            : "⚠ Episode timed out after 20 steps. Resetting…",
          head:      result.selected_head,
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, sys]);

        // New episode after 2 s (only if auto mode)
        if (isAutoMode) {
          setTimeout(() => {
            setStudentState(INITIAL_STATE);
            setEpisodeActive(true);
            resetEpisode();
          }, 2000);
        }
      }

      setTrainingStats(dqnClient.getTrainingStats());
      setLastQValues(result.q_values);
    } catch (e: any) {
      if (e.name === 'AbortError') return;
      console.error(e);
    } finally {
      setIsThinking(false);
    }
  }, [episodeActive, sessionId, isThinking]);

  // Schedule next auto-step when state changes + episode is active
  useEffect(() => {
    if (!episodeActive || !isAutoMode) return;
    stepTimer.current = setTimeout(() => fireStep(studentState), 3000);
    return () => { if (stepTimer.current) clearTimeout(stepTimer.current); };
  }, [studentState, episodeActive, isAutoMode, fireStep]);

  // Handle manual student input
  const handleUserMessage = useCallback(() => {
    if (!userInput.trim() || isThinking || !episodeActive) return;
    
    // Add student message to feed
    const studentMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "student",
      content: userInput,
      timestamp: Date.now(),
    };
    
    setMessages(prev => [...prev, studentMsg]);
    setUserInput("");
    
    // Trigger tutor's next step based on the current state
    fireStep(studentState, userInput);
  }, [userInput, isThinking, episodeActive, studentState, fireStep]);

  // What-If handler
  const handleWhatIf = useCallback(async (overrideState: StudentState): Promise<WhatIfResult> => {
    const result = await dqnClient.forward(overrideState, undefined, undefined, sessionId);
    return {
      predicted_action: result.action as ActionType,
      selected_head:    result.selected_head,
      q_values:         result.q_values as Record<ActionType, number>,
      expected_reward:  result.expected_reward,
      confusion_delta:  result.confusion_delta,
      attention_delta:  result.attention_delta,
    };
  }, []);

  const confusion = studentState.confusion;
  const confColor = confusion <= 2 ? "#5bf587" : confusion <= 5 ? "#f5e45b" : "#cf4a2a";

  return (
    <div
      style={{
        display:         "grid",
        gridTemplateColumns: "260px 1fr 320px",
        gridTemplateRows: "48px 1fr",
        height:          "100%",
        width:           "100%",
        background:      "#050810",
        fontFamily:      "'JetBrains Mono', 'Fira Code', monospace",
        overflow:        "hidden",
        color:           "#c8dce8",
      }}
    >
      {/* ── Top bar ── */}
      <header
        style={{
          gridColumn:    "1 / -1",
          display:       "flex",
          alignItems:    "center",
          gap:           "16px",
          padding:       "0 20px",
          background:    "#080c10",
          borderBottom:  "1px solid #1a2a36",
          zIndex:        10,
        }}
      >
        <span style={{ fontSize: "11px", fontWeight: 700, color: "#5bc4f5", letterSpacing: "0.18em" }}>
          EDUFORGE
        </span>
        <span style={{ color: "#1a2a36" }}>|</span>
        <span style={{ fontSize: "10px", color: "#3a5060" }}>Multi-Head DQN Tutor</span>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "16px" }}>
          {/* Live confusion gauge */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "9px", color: "#3a5060", letterSpacing: "0.1em" }}>CONFUSION</span>
            <div style={{ width: "80px", height: "4px", background: "#1a2a36", borderRadius: "2px" }}>
              <div
                style={{
                  height: "100%",
                  width:  `${(confusion / 10) * 100}%`,
                  background: confColor,
                  borderRadius: "2px",
                  transition: "width 0.4s ease, background 0.4s ease",
                }}
              />
            </div>
            <span style={{ fontSize: "10px", color: confColor, fontWeight: 700 }}>
              {confusion.toFixed(1)}
            </span>
          </div>

          {/* Attention */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "9px", color: "#3a5060", letterSpacing: "0.1em" }}>ATTENTION</span>
            <span style={{ fontSize: "10px", color: "#5bc4f5", fontWeight: 700 }}>
              {studentState.attention.toFixed(1)}
            </span>
          </div>

          {/* Head badge */}
          <HeadBadge head={activeHead} />

          {/* Auto Mode Toggle */}
          <button
            onClick={() => setIsAutoMode(!isAutoMode)}
            style={{
              background: isAutoMode ? "#1e8a3a33" : "#cf4a2a33",
              border: `1px solid ${isAutoMode ? "#1e8a3a" : "#cf4a2a"}`,
              color: isAutoMode ? "#5bf587" : "#cf4a2a",
              fontSize: "9px",
              fontWeight: 700,
              padding: "4px 8px",
              borderRadius: "4px",
              cursor: "pointer",
              letterSpacing: "0.05em",
            }}
          >
            {isAutoMode ? "🤖 AUTO-PILOT" : "👤 USER MODE"}
          </button>
          
          {/* Manual Step Button (only visible in user mode) */}
          {!isAutoMode && episodeActive && (
            <button
              onClick={() => fireStep(studentState)}
              disabled={isThinking}
              style={{
                background: "#1e6fa533",
                border: "1px solid #1e6fa5",
                color: "#5bc4f5",
                fontSize: "9px",
                fontWeight: 700,
                padding: "4px 8px",
                borderRadius: "4px",
                cursor: isThinking ? "not-allowed" : "pointer",
              }}
            >
              NEXT STEP →
            </button>
          )}
          
          {!isAutoMode && !episodeActive && (
            <button
              onClick={() => {
                setStudentState(INITIAL_STATE);
                setEpisodeActive(true);
                resetEpisode();
              }}
              style={{
                background: "#1e8a3a33",
                border: "1px solid #1e8a3a",
                color: "#5bf587",
                fontSize: "9px",
                fontWeight: 700,
                padding: "4px 8px",
                borderRadius: "4px",
                cursor: "pointer",
              }}
            >
              START NEW EPISODE
            </button>
          )}

          {/* Episode indicator */}
          <div
            style={{
              display:    "flex",
              alignItems: "center",
              gap:        "5px",
              fontSize:   "9px",
              color:      episodeActive ? "#5bf587" : "#3a5060",
            }}
          >
            <span
              style={{
                width: "6px", height: "6px", borderRadius: "50%",
                background: episodeActive ? "#5bf587" : "#3a5060",
                animation:  episodeActive ? "blink 1.4s ease-in-out infinite" : "none",
              }}
            />
            {episodeActive ? "EPISODE ACTIVE" : "RESETTING"}
            <style>{`@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}`}</style>
          </div>
        </div>
      </header>

      {/* ── Left sidebar ── */}
      <aside
        style={{
          display:       "flex",
          flexDirection: "column",
          borderRight:   "1px solid #1a2a36",
          background:    "#080c10",
          overflow:      "hidden",
        }}
      >
        {/* Sidebar tabs */}
        <div
          style={{
            display:       "grid",
            gridTemplateColumns: "1fr 1fr",
            borderBottom:  "1px solid #1a2a36",
          }}
        >
          {(["state", "sim"] as SidebarTab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setSidebarTab(tab)}
              style={{
                background:  sidebarTab === tab ? "#0d1820" : "transparent",
                border:      "none",
                borderBottom: sidebarTab === tab ? "2px solid #5bc4f5" : "2px solid transparent",
                color:       sidebarTab === tab ? "#5bc4f5" : "#3a5060",
                fontSize:    "9px",
                fontWeight:  700,
                letterSpacing: "0.1em",
                padding:     "10px",
                cursor:      "pointer",
                fontFamily:  "inherit",
              }}
            >
              {tab === "state" ? "STATE" : "WHAT-IF"}
            </button>
          ))}
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: "12px" }}>
          {sidebarTab === "state" ? (
            <StatePanel state={studentState} />
          ) : (
            <SimulatorInput onRun={handleWhatIf} />
          )}
        </div>
      </aside>

      {/* ── Chat feed (center) ── */}
      <main style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <ChatFeed messages={messages} isThinking={isThinking} />
        
        {/* User Mode Input */}
        {!isAutoMode && episodeActive && (
          <div style={{
            padding: "16px",
            background: "#080c10",
            borderTop: "1px solid #1a2a36",
            display: "flex",
            gap: "12px",
            alignItems: "center"
          }}>
            <input 
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="Type your student response or question..."
              onKeyDown={(e) => {
                 if (e.key === 'Enter') handleUserMessage();
              }}
              style={{
                flex: 1,
                background: "#111820",
                border: "1px solid #1a2a36",
                color: "#c8dce8",
                padding: "10px 14px",
                borderRadius: "6px",
                fontFamily: "inherit",
                fontSize: "12px",
                outline: "none"
              }}
              disabled={isThinking}
            />
            {isThinking ? (
              <button 
                onClick={() => {
                  dqnClient.abortCurrent();
                  setIsThinking(false);
                }}
                style={{
                  background: "#f43f5e",
                  color: "#fff",
                  border: "none",
                  padding: "10px 16px",
                  borderRadius: "6px",
                  cursor: "pointer",
                  fontWeight: 700,
                  fontSize: "12px",
                  fontFamily: "inherit"
                }}
              >
                Cancel ✕
              </button>
            ) : (
              <button 
                 onClick={handleUserMessage}
                 disabled={!userInput.trim()}
                 style={{
                   background: !userInput.trim() ? "#1e6fa566" : "#1e6fa5",
                   color: !userInput.trim() ? "#5bc4f588" : "#fff",
                   border: "none",
                   padding: "10px 16px",
                   borderRadius: "6px",
                   cursor: !userInput.trim() ? "not-allowed" : "pointer",
                   fontWeight: 700,
                   fontSize: "12px",
                   fontFamily: "inherit"
                 }}
              >
                 Send →
              </button>
            )}
          </div>
        )}
      </main>

      {/* ── Right analytics panel ── */}
      <aside
        style={{
          display:       "flex",
          flexDirection: "column",
          borderLeft:    "1px solid #1a2a36",
          background:    "#080c10",
          overflow:      "hidden",
        }}
      >
        <div
          style={{
            padding:      "10px 14px",
            borderBottom: "1px solid #1a2a36",
            fontSize:     "9px",
            fontWeight:   700,
            color:        "#3a5060",
            letterSpacing: "0.12em",
          }}
        >
          ANALYTICS · HEAD PERFORMANCE
        </div>
        <div style={{ flex: 1, overflowY: "auto", padding: "12px" }}>
          <HeadSwitcher
            stats={trainingStats}
            activeHead={activeHead}
            onHeadChange={setActiveHead}
          />
          <div style={{ marginTop: "16px" }}>
            <ActionPanel 
              qValues={lastQValues} 
              onAction={(action) => fireStep(studentState, undefined, action)}
              disabled={isThinking || !episodeActive}
            />
          </div>
        </div>
      </aside>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function HeadBadge({ head }: { head: HeadType }) {
  const COLORS: Record<HeadType, { text: string; bg: string; border: string }> = {
    FACTUAL:    { text: "#5bc4f5", bg: "#1e6fa518", border: "#1e6fa544" },
    PROCEDURAL: { text: "#f5e45b", bg: "#c4a81e18", border: "#c4a81e44" },
    CONCEPTUAL: { text: "#e87af5", bg: "#b42ecf18", border: "#b42ecf44" },
    TRANSFER:   { text: "#5bf587", bg: "#2acf5a18", border: "#2acf5a44" },
  };
  const c = COLORS[head];
  return (
    <span
      style={{
        fontSize:     "9px",
        fontWeight:   700,
        letterSpacing: "0.1em",
        color:        c.text,
        background:   c.bg,
        border:       `1px solid ${c.border}`,
        padding:      "2px 8px",
        borderRadius: "3px",
      }}
    >
      {head}
    </span>
  );
}

function StatePanel({ state }: { state: StudentState }) {
  const DOMAIN_LABELS: Record<number, HeadType> = {
    0: "FACTUAL", 1: "PROCEDURAL", 2: "CONCEPTUAL", 3: "TRANSFER",
  };

  const rows: { label: string; value: string; color?: string }[] = [
    { label: "confusion",          value: state.confusion.toFixed(2),
      color: state.confusion > 7 ? "#cf4a2a" : state.confusion > 4 ? "#f5e45b" : "#5bf587" },
    { label: "attention",          value: state.attention.toFixed(2),
      color: state.attention < 4 ? "#cf4a2a" : "#5bc4f5" },
    { label: "domain",             value: DOMAIN_LABELS[state.domain_index] },
    { label: "step_count",         value: String(state.step_count) },
    { label: "last_action_impact", value: state.last_action_impact.toFixed(3),
      color: state.last_action_impact >= 0 ? "#5bf587" : "#cf4a2a" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      <div style={{ fontSize: "9px", color: "#3a5060", letterSpacing: "0.1em", marginBottom: "4px" }}>
        LIVE STUDENT STATE
      </div>
      {rows.map(({ label, value, color }) => (
        <div
          key={label}
          style={{
            display:        "flex",
            justifyContent: "space-between",
            alignItems:     "center",
            padding:        "7px 10px",
            background:     "#0d1820",
            border:         "1px solid #1a2a36",
            borderRadius:   "5px",
          }}
        >
          <span style={{ fontSize: "9px", color: "#3a5060" }}>{label}</span>
          <span style={{ fontSize: "11px", fontWeight: 700, color: color ?? "#9ab8cc" }}>{value}</span>
        </div>
      ))}

      {/* Confusion bar */}
      <div style={{ marginTop: "8px" }}>
        <div
          style={{
            fontSize: "9px", color: "#3a5060", letterSpacing: "0.1em", marginBottom: "5px",
          }}
        >
          CONFUSION TRAJECTORY
        </div>
        <div
          style={{
            height:     "8px",
            background: "#1a2a36",
            borderRadius: "4px",
            overflow:   "hidden",
          }}
        >
          <div
            style={{
              height:     "100%",
              width:      `${(state.confusion / 10) * 100}%`,
              background: state.confusion > 7
                ? "linear-gradient(90deg, #cf4a2a, #f57a2a)"
                : state.confusion > 4
                ? "linear-gradient(90deg, #c4a81e, #f5e45b)"
                : "linear-gradient(90deg, #1e8a3a, #5bf587)",
              borderRadius: "4px",
              transition:  "width 0.4s ease, background 0.4s ease",
            }}
          />
        </div>
        <div
          style={{
            display:        "flex",
            justifyContent: "space-between",
            fontSize:       "9px",
            color:          "#2a3a44",
            marginTop:      "3px",
          }}
        >
          <span>0 (clear)</span>
          <span>10 (max confusion)</span>
        </div>
      </div>

      {/* Step progress */}
      <div style={{ marginTop: "4px" }}>
        <div
          style={{ fontSize: "9px", color: "#3a5060", letterSpacing: "0.1em", marginBottom: "5px" }}
        >
          EPISODE PROGRESS
        </div>
        <div
          style={{
            height: "4px", background: "#1a2a36", borderRadius: "2px", overflow: "hidden",
          }}
        >
          <div
            style={{
              height:     "100%",
              width:      `${(state.step_count / 20) * 100}%`,
              background: state.step_count > 15 ? "#cf4a2a" : "#5bc4f5",
              borderRadius: "2px",
              transition:  "width 0.3s ease",
            }}
          />
        </div>
        <div
          style={{
            textAlign: "right", fontSize: "9px", color: "#2a3a44", marginTop: "3px",
          }}
        >
          {state.step_count} / 20 steps
        </div>
      </div>
    </div>
  );
}
