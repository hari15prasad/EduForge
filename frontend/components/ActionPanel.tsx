"use client";

import React, { useState, useCallback, useMemo } from "react";
import { useRL, Action } from "@/components/RLProvider";

// ─── Types ───────────────────────────────────────────────────────────────────

interface ActionMeta {
  id: Action;
  label: string;
  shortLabel: string;
  icon: string;
  description: string;
  keybind: string;
  accentColor: string;
  glowColor: string;
}

// ─── Action Config ────────────────────────────────────────────────────────────

const ACTIONS: ActionMeta[] = [
  {
    id: "explain",
    label: "Explain",
    shortLabel: "EXP",
    icon: "◈",
    description: "Deliver a clear conceptual explanation",
    keybind: "1",
    accentColor: "#6366f1",
    glowColor: "rgba(99,102,241,0.45)",
  },
  {
    id: "correct_fact",
    label: "Correct Fact",
    shortLabel: "COR",
    icon: "⊕",
    description: "Address and correct a factual misconception",
    keybind: "2",
    accentColor: "#22d3ee",
    glowColor: "rgba(34,211,238,0.45)",
  },
  {
    id: "worked_example",
    label: "Worked Example",
    shortLabel: "WEX",
    icon: "▦",
    description: "Walk through a step-by-step solved problem",
    keybind: "3",
    accentColor: "#10b981",
    glowColor: "rgba(16,185,129,0.45)",
  },
  {
    id: "analogize",
    label: "Analogize",
    shortLabel: "ANA",
    icon: "⟷",
    description: "Draw a bridging analogy to prior knowledge",
    keybind: "4",
    accentColor: "#8b5cf6",
    glowColor: "rgba(139,92,246,0.45)",
  },
  {
    id: "question",
    label: "Question",
    shortLabel: "QST",
    icon: "⊙",
    description: "Pose a Socratic probe to test understanding",
    keybind: "5",
    accentColor: "#f59e0b",
    glowColor: "rgba(245,158,11,0.45)",
  },
];

// ─── Mock Q-value generator ───────────────────────────────────────────────────

function mockQValues(domainIndex: number, confusion: number, attention: number): Record<Action, number> {
  // Simulate domain-influenced Q-values (range roughly -5 to +15)
  const base: Record<Action, number> = {
    explain:       8.2 - domainIndex * 0.8 + (10 - confusion) * 0.4,
    correct_fact:  6.5 + (domainIndex === 0 ? 3 : 0) - confusion * 0.2,
    worked_example: 9.1 + (domainIndex === 1 ? 2.5 : 0) - domainIndex * 0.5,
    analogize:     5.8 + (domainIndex >= 2 ? 4 : 0) + attention * 0.3,
    question:      4.2 + attention * 0.6 - (10 - attention) * 0.1,
  };
  // Removed Math.random() to prevent SSR hydration mismatch
  return Object.fromEntries(
    Object.entries(base).map(([k, v]) => [k, +(v).toFixed(2)])
  ) as Record<Action, number>;
}

function qColor(q: number): string {
  if (q >= 8) return "#10b981";
  if (q >= 4) return "#f59e0b";
  if (q >= 0) return "#8899cc";
  return "#f43f5e";
}

// ─── Action Button ────────────────────────────────────────────────────────────

interface ActionButtonProps {
  meta: ActionMeta;
  qValue: number;
  isBest: boolean;
  isActive: boolean;
  disabled: boolean;
  onClick: (action: Action) => void;
}

function ActionButton({ meta, qValue, isBest, isActive, disabled, onClick }: ActionButtonProps) {
  const [pressed, setPressed] = useState(false);

  const handleClick = useCallback(() => {
    if (disabled) return;
    setPressed(true);
    setTimeout(() => setPressed(false), 300);
    onClick(meta.id);
  }, [disabled, meta.id, onClick]);

  return (
    <button
      className={`action-btn ${isBest ? "action-btn--best" : ""} ${isActive ? "action-btn--active" : ""} ${pressed ? "action-btn--pressed" : ""} ${disabled ? "action-btn--disabled" : ""}`}
      onClick={handleClick}
      disabled={disabled}
      aria-label={`${meta.label}: Q=${qValue}`}
      style={{
        "--accent": meta.accentColor,
        "--glow": meta.glowColor,
      } as React.CSSProperties}
    >
      {isBest && <span className="action-btn__best-tag">OPTIMAL</span>}

      <div className="action-btn__top">
        <span className="action-btn__icon">{meta.icon}</span>
        <span className="action-btn__keybind">[{meta.keybind}]</span>
      </div>

      <div className="action-btn__middle">
        <span className="action-btn__label">{meta.label}</span>
        <span className="action-btn__short">{meta.shortLabel}</span>
      </div>

      <p className="action-btn__desc">{meta.description}</p>

      <div className="action-btn__q">
        <span className="action-btn__q-label">Q-VAL</span>
        <span
          className="action-btn__q-value"
          style={{ color: qColor(qValue || 0) }}
        >
          {(qValue || 0) > 0 ? "+" : ""}{(qValue || 0).toFixed(2)}
        </span>
      </div>

      {isActive && <span className="action-btn__ripple" />}
    </button>
  );
}

// ─── Action Panel ─────────────────────────────────────────────────────────────

interface ActionPanelProps {
  qValues?: Record<string, number>;
  onAction?: (action: Action) => void;
  disabled?: boolean;
}

export default function ActionPanel({ qValues: externalQValues, onAction, disabled: externalDisabled }: ActionPanelProps) {
  const { state, logAction, setAgentStatus } = useRL();
  const [lastAction, setLastAction] = useState<Action | null>(null);

  const mockQ = useMemo(() => mockQValues(state.domainIndex, state.confusion, state.attention), [state.domainIndex, state.confusion, state.attention]);
  const qValues = (externalQValues && Object.keys(externalQValues).length > 0) 
    ? externalQValues 
    : mockQ;
  const bestAction = (Object.entries(qValues) as [Action, number][]).reduce(
    (best, [a, q]) => (q > best[1] ? [a, q] : best),
    ["explain", -Infinity] as [Action, number]
  )[0];

  const handleAction = useCallback(
    (action: Action) => {
      if (onAction) {
        onAction(action);
      } else {
        logAction(action, state.currentStep);
      }
      setLastAction(action);
      setAgentStatus("training");
      setTimeout(() => setAgentStatus("online"), 800);
    },
    [logAction, state.currentStep, setAgentStatus, onAction]
  );

  const disabled = externalDisabled ?? state.isTerminated;

  return (
    <section className="action-panel">
      <header className="action-panel__header">
        <div>
          <span className="action-panel__title">TACTICAL ACTIONS</span>
          <span className="action-panel__subtitle">DQN Policy Head · Step {state.currentStep}</span>
        </div>
        {lastAction && (
          <div className="action-panel__last">
            <span className="action-panel__last-label">Last</span>
            <span className="action-panel__last-value">
              {ACTIONS.find((a) => a.id === lastAction)?.shortLabel ?? lastAction}
            </span>
          </div>
        )}
      </header>

      <div className="action-panel__grid">
        {ACTIONS.map((meta) => (
          <ActionButton
            key={meta.id}
            meta={meta}
            qValue={qValues[meta.id]}
            isBest={meta.id === bestAction}
            isActive={meta.id === lastAction}
            disabled={disabled}
            onClick={handleAction}
          />
        ))}
      </div>

      {disabled && (
        <div className="action-panel__terminated">
          Episode Terminated — {state.confusion <= 2 ? "✓ SUCCESS" : "✗ TIMEOUT"}
        </div>
      )}

      <style>{`
        .action-panel {
          background: rgba(10,14,26,0.6);
          border: 1px solid #1e2d50;
          border-radius: 20px;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .action-panel__header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
        }
        .action-panel__title {
          display: block;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          letter-spacing: 0.16em;
          color: #4a5580;
        }
        .action-panel__subtitle {
          display: block;
          font-family: 'DM Sans', system-ui, sans-serif;
          font-size: 12px;
          color: #8899cc;
          margin-top: 2px;
        }
        .action-panel__last {
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .action-panel__last-label {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          color: #4a5580;
        }
        .action-panel__last-value {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          color: #6366f1;
          padding: 2px 8px;
          border: 1px solid #4338ca;
          border-radius: 4px;
          background: rgba(99,102,241,0.1);
        }
        .action-panel__grid {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        /* Action button */
        .action-btn {
          position: relative;
          display: flex;
          flex-direction: column;
          gap: 8px;
          padding: 12px 10px;
          background: rgba(15,22,41,0.8);
          border: 1px solid #1e2d50;
          border-radius: 12px;
          cursor: pointer;
          text-align: left;
          transition: border-color 200ms ease, box-shadow 200ms ease, transform 120ms ease, background 200ms ease;
          overflow: hidden;
        }
        .action-btn:hover:not(.action-btn--disabled) {
          border-color: var(--accent);
          box-shadow: 0 0 16px var(--glow), 0 4px 20px rgba(0,0,0,0.4);
          background: rgba(21,30,56,0.9);
          transform: translateY(-2px);
        }
        .action-btn--best {
          border-color: var(--accent);
          box-shadow: 0 0 12px var(--glow);
          background: rgba(21,30,56,0.9);
        }
        .action-btn--active {
          background: rgba(21,30,56,1);
          transform: scale(0.98);
        }
        .action-btn--pressed {
          transform: scale(0.96);
        }
        .action-btn--disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .action-btn__best-tag {
          position: absolute;
          top: 6px;
          right: 6px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 8px;
          letter-spacing: 0.1em;
          padding: 1px 5px;
          border-radius: 3px;
          background: var(--accent);
          color: #0a0e1a;
          font-weight: 700;
        }

        .action-btn__top {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }
        .action-btn__icon {
          font-size: 20px;
          color: var(--accent);
          line-height: 1;
          filter: drop-shadow(0 0 4px var(--glow));
        }
        .action-btn__keybind {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          color: #4a5580;
        }

        .action-btn__middle {
          display: flex;
          flex-direction: column;
          gap: 1px;
        }
        .action-btn__label {
          font-family: 'DM Sans', system-ui, sans-serif;
          font-size: 12px;
          font-weight: 600;
          color: #e8eeff;
          line-height: 1.2;
        }
        .action-btn__short {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          color: var(--accent);
          letter-spacing: 0.08em;
        }

        .action-btn__desc {
          font-family: 'DM Sans', system-ui, sans-serif;
          font-size: 10px;
          color: #4a5580;
          line-height: 1.4;
          flex: 1;
        }

        .action-btn__q {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 5px 6px;
          background: rgba(10,14,26,0.6);
          border-radius: 6px;
          border: 1px solid #131d38;
        }
        .action-btn__q-label {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          color: #4a5580;
          letter-spacing: 0.1em;
        }
        .action-btn__q-value {
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          font-weight: 700;
          transition: color 300ms ease;
        }

        .action-btn__ripple {
          position: absolute;
          inset: 0;
          background: var(--accent);
          opacity: 0;
          border-radius: inherit;
          animation: rippleFade 400ms ease forwards;
        }
        @keyframes rippleFade {
          0%   { opacity: 0.12; transform: scale(0.9); }
          100% { opacity: 0;    transform: scale(1.05); }
        }

        .action-panel__terminated {
          text-align: center;
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          color: #f59e0b;
          padding: 8px;
          border: 1px solid rgba(245,158,11,0.3);
          border-radius: 8px;
          background: rgba(245,158,11,0.06);
          letter-spacing: 0.08em;
        }
      `}</style>
    </section>
  );
}
