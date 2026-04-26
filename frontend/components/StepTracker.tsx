"use client";

import React, { useMemo } from "react";
import { useRL } from "@/components/RLProvider";

// ─── Constants ────────────────────────────────────────────────────────────────

const MAX_STEPS = 20;
const DANGER_THRESHOLD = 18;   // neon-red kicks in at step 18+
const WARNING_THRESHOLD = 14;  // amber caution zone

// ─── Helpers ─────────────────────────────────────────────────────────────────

type StepTier = "safe" | "caution" | "danger";

function getTier(step: number): StepTier {
  if (step >= DANGER_THRESHOLD) return "danger";
  if (step >= WARNING_THRESHOLD) return "caution";
  return "safe";
}

const TIER_STYLES: Record<StepTier, { bar: string; glow: string; text: string; label: string }> = {
  safe:    { bar: "#6366f1", glow: "rgba(99,102,241,0.5)",  text: "#818cf8", label: "RUNNING"         },
  caution: { bar: "#f59e0b", glow: "rgba(245,158,11,0.5)",  text: "#fbbf24", label: "CAUTION"          },
  danger:  { bar: "#f43f5e", glow: "rgba(244,63,94,0.65)",  text: "#fb7185", label: "TIMEOUT IMMINENT" },
};

// ─── Step Pip ─────────────────────────────────────────────────────────────────

interface StepPipProps {
  index: number;       // 1-based display index
  filled: boolean;
  active: boolean;
  tier: StepTier;
  isDanger: boolean;
}

function StepPip({ index, filled, active, tier, isDanger }: StepPipProps) {
  const { bar, glow } = TIER_STYLES[tier];
  const pipColor = filled ? (isDanger ? "#f43f5e" : bar) : "#1a2540";

  return (
    <div
      className={`pip ${filled ? "pip--filled" : ""} ${active ? "pip--active" : ""} ${isDanger && filled ? "pip--danger" : ""}`}
      title={`Step ${index}`}
      style={{
        "--pip-color": pipColor,
        "--pip-glow": glow,
      } as React.CSSProperties}
    >
      {index % 5 === 0 && (
        <span className="pip__label">{index}</span>
      )}
    </div>
  );
}

// ─── Step Tracker ─────────────────────────────────────────────────────────────

export default function StepTracker() {
  const { state } = useRL();
  const { currentStep, isTerminated, confusion } = state;

  const tier = getTier(currentStep);
  const { bar, glow, text, label } = TIER_STYLES[tier];
  const pct = Math.min(100, (currentStep / MAX_STEPS) * 100);

  const stepsRemaining = MAX_STEPS - currentStep;
  const outcome = isTerminated
    ? confusion <= 2.0
      ? { tag: "SUCCESS", color: "#10b981", glow: "rgba(16,185,129,0.5)" }
      : { tag: "TIMEOUT", color: "#f43f5e", glow: "rgba(244,63,94,0.5)" }
    : null;

  const pips = useMemo(
    () =>
      Array.from({ length: MAX_STEPS }, (_, i) => ({
        index: i + 1,
        filled: i < currentStep,
        active: i === currentStep - 1,
        tier: getTier(i + 1),
        isDanger: i + 1 >= DANGER_THRESHOLD,
      })),
    [currentStep]
  );

  return (
    <section
      className={`step-tracker ${tier === "danger" ? "step-tracker--danger" : ""}`}
      style={{ "--bar": bar, "--glow": glow, "--text": text } as React.CSSProperties}
    >
      {/* Header row */}
      <div className="st-header">
        <div className="st-header__left">
          <span className="st-title">EPISODE PROGRESS</span>
          <span className="st-subtitle">Markov Decision Steps</span>
        </div>

        <div className="st-header__right">
          {outcome ? (
            <span
              className="st-outcome"
              style={{ color: outcome.color, borderColor: outcome.color, boxShadow: `0 0 10px ${outcome.glow}` }}
            >
              {outcome.tag}
            </span>
          ) : (
            <span className="st-status" style={{ color: text }}>
              {label}
            </span>
          )}
        </div>
      </div>

      {/* Main progress bar */}
      <div className="st-bar-wrap">
        <div className="st-bar-track">
          {/* Segmented danger zone overlay */}
          <div
            className="st-bar-danger-zone"
            style={{ left: `${(DANGER_THRESHOLD / MAX_STEPS) * 100}%` }}
          />
          {/* Fill */}
          <div
            className={`st-bar-fill ${tier === "danger" ? "st-bar-fill--pulse" : ""}`}
            style={{ width: `${pct}%`, background: bar, boxShadow: `0 0 12px ${glow}` }}
          />
          {/* Warning marker at step 18 */}
          <div
            className="st-bar-marker"
            style={{ left: `${(DANGER_THRESHOLD / MAX_STEPS) * 100}%` }}
            title={`Danger at step ${DANGER_THRESHOLD}`}
          />
        </div>

        {/* Step labels beneath bar */}
        <div className="st-bar-labels">
          <span>0</span>
          <span style={{ position: "absolute", left: `${(DANGER_THRESHOLD / MAX_STEPS) * 100}%`, transform: "translateX(-50%)", color: "#f43f5e" }}>
            {DANGER_THRESHOLD}
          </span>
          <span>20</span>
        </div>
      </div>

      {/* Pip grid */}
      <div className="st-pips">
        {pips.map((p) => (
          <StepPip key={p.index} {...p} />
        ))}
      </div>

      {/* Footer stats */}
      <div className="st-footer">
        <div className="st-stat">
          <span className="st-stat__label">Current Step</span>
          <span className="st-stat__value" style={{ color: text }}>{currentStep}</span>
        </div>
        <div className="st-stat">
          <span className="st-stat__label">Remaining</span>
          <span
            className="st-stat__value"
            style={{ color: stepsRemaining <= 2 ? "#f43f5e" : "#e8eeff" }}
          >
            {stepsRemaining}
          </span>
        </div>
        <div className="st-stat">
          <span className="st-stat__label">Danger At</span>
          <span className="st-stat__value" style={{ color: "#f43f5e" }}>{DANGER_THRESHOLD}</span>
        </div>
        <div className="st-stat">
          <span className="st-stat__label">Confusion</span>
          <span
            className="st-stat__value"
            style={{ color: confusion <= 2 ? "#10b981" : confusion >= 7 ? "#f43f5e" : "#f59e0b" }}
          >
            {confusion.toFixed(1)}
          </span>
        </div>
      </div>

      {/* Danger flash overlay */}
      {tier === "danger" && !isTerminated && (
        <div className="st-danger-overlay" aria-hidden="true" />
      )}

      <style>{`
        .step-tracker {
          position: relative;
          background: rgba(10,14,26,0.6);
          border: 1px solid #1e2d50;
          border-radius: 20px;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 14px;
          overflow: hidden;
          transition: border-color 400ms ease, box-shadow 400ms ease;
        }
        .step-tracker--danger {
          border-color: rgba(244,63,94,0.5);
          box-shadow: 0 0 24px rgba(244,63,94,0.2), inset 0 0 40px rgba(244,63,94,0.04);
        }

        /* Header */
        .st-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
        }
        .st-header__left { display: flex; flex-direction: column; gap: 2px; }
        .st-title {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          letter-spacing: 0.16em;
          color: #4a5580;
        }
        .st-subtitle {
          font-family: 'DM Sans', system-ui, sans-serif;
          font-size: 11px;
          color: #8899cc;
        }
        .st-header__right { display: flex; align-items: center; }
        .st-status {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          letter-spacing: 0.1em;
        }
        .st-outcome {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          letter-spacing: 0.12em;
          padding: 3px 10px;
          border: 1px solid;
          border-radius: 5px;
        }

        /* Progress bar */
        .st-bar-wrap { display: flex; flex-direction: column; gap: 4px; }
        .st-bar-track {
          position: relative;
          height: 8px;
          background: #1a2540;
          border-radius: 4px;
          overflow: visible;
        }
        .st-bar-fill {
          position: absolute;
          top: 0; left: 0; bottom: 0;
          border-radius: 4px;
          transition: width 300ms cubic-bezier(0.4,0,0.2,1), background 400ms ease, box-shadow 400ms ease;
        }
        .st-bar-fill--pulse {
          animation: barPulse 800ms ease-in-out infinite;
        }
        @keyframes barPulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.7; }
        }
        .st-bar-danger-zone {
          position: absolute;
          top: -3px;
          right: 0;
          bottom: -3px;
          background: rgba(244,63,94,0.06);
          border-left: 1px dashed rgba(244,63,94,0.3);
          border-radius: 0 4px 4px 0;
        }
        .st-bar-marker {
          position: absolute;
          top: -6px;
          bottom: -6px;
          width: 2px;
          background: rgba(244,63,94,0.6);
          border-radius: 1px;
          box-shadow: 0 0 6px rgba(244,63,94,0.6);
        }
        .st-bar-labels {
          position: relative;
          display: flex;
          justify-content: space-between;
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          color: #4a5580;
        }

        /* Pip grid */
        .st-pips {
          display: grid;
          grid-template-columns: repeat(20, 1fr);
          gap: 3px;
        }

        .pip {
          position: relative;
          height: 20px;
          background: var(--pip-color);
          border-radius: 3px;
          transition: background 300ms ease, box-shadow 300ms ease, transform 150ms ease;
          cursor: default;
        }
        .pip--filled {
          box-shadow: 0 0 4px var(--pip-glow, transparent);
        }
        .pip--active {
          transform: scaleY(1.3);
          box-shadow: 0 0 8px var(--pip-glow, transparent);
        }
        .pip--danger.pip--filled {
          animation: pipDangerPulse 600ms ease-in-out infinite;
        }
        @keyframes pipDangerPulse {
          0%, 100% { box-shadow: 0 0 4px rgba(244,63,94,0.5); }
          50%       { box-shadow: 0 0 10px rgba(244,63,94,0.9); }
        }
        .pip__label {
          position: absolute;
          bottom: -14px;
          left: 50%;
          transform: translateX(-50%);
          font-family: 'JetBrains Mono', monospace;
          font-size: 8px;
          color: #4a5580;
          pointer-events: none;
        }

        /* Footer */
        .st-footer {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 8px;
          margin-top: 8px; /* extra room for pip labels */
        }
        .st-stat {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 3px;
          padding: 8px;
          background: rgba(15,22,41,0.8);
          border: 1px solid #1e2d50;
          border-radius: 8px;
        }
        .st-stat__label {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          color: #4a5580;
          letter-spacing: 0.08em;
        }
        .st-stat__value {
          font-family: 'JetBrains Mono', monospace;
          font-size: 16px;
          font-weight: 700;
          transition: color 400ms ease;
        }

        /* Danger overlay */
        .st-danger-overlay {
          position: absolute;
          inset: 0;
          pointer-events: none;
          background: linear-gradient(135deg, transparent 70%, rgba(244,63,94,0.04) 100%);
          animation: overlayPulse 1.2s ease-in-out infinite;
          border-radius: inherit;
        }
        @keyframes overlayPulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0; }
        }
      `}</style>
    </section>
  );
}
