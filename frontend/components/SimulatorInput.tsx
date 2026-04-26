import { useState } from "react";

export type HeadType = "FACTUAL" | "PROCEDURAL" | "CONCEPTUAL" | "TRANSFER";
export type ActionType = "explain" | "correct_fact" | "worked_example" | "analogize" | "question";

export interface StudentState {
  confusion: number;       // 0–10
  attention: number;       // 0–10
  domain_index: number;    // 0–3
  step_count: number;      // 0–20
  last_action_impact: number; // -1 to 1
}

export interface WhatIfResult {
  predicted_action: ActionType;
  selected_head: HeadType;
  q_values: Record<ActionType, number>;
  expected_reward: number;
  confusion_delta: number;
  attention_delta: number;
}

interface SimulatorInputProps {
  onRun: (state: StudentState) => Promise<WhatIfResult>;
  isRunning?: boolean;
}

const DOMAIN_LABELS: Record<number, HeadType> = {
  0: "FACTUAL",
  1: "PROCEDURAL",
  2: "CONCEPTUAL",
  3: "TRANSFER",
};

const ACTION_COLORS: Record<ActionType, string> = {
  explain:        "#5bc4f5",
  correct_fact:   "#5bf587",
  worked_example: "#f5e45b",
  analogize:      "#e87af5",
  question:       "#f59a5b",
};

const DEFAULTS: StudentState = {
  confusion: 6.0,
  attention: 7.0,
  domain_index: 0,
  step_count: 0,
  last_action_impact: 0.0,
};

export default function SimulatorInput({ onRun, isRunning = false }: SimulatorInputProps) {
  const [state, setState] = useState<StudentState>(DEFAULTS);
  const [result, setResult] = useState<WhatIfResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const set = (key: keyof StudentState, value: number) =>
    setState((prev) => ({ ...prev, [key]: value }));

  const handleRun = async () => {
    setError(null);
    setResult(null);
    try {
      const res = await onRun(state);
      setResult(res);
    } catch (e: any) {
      setError(e?.message ?? "Unknown error");
    }
  };

  const handleReset = () => {
    setState(DEFAULTS);
    setResult(null);
    setError(null);
  };

  return (
    <div
      style={{
        background: "#080c10",
        border: "1px solid #1a2a36",
        borderRadius: "10px",
        overflow: "hidden",
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid #1a2a36",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "#0a1018",
        }}
      >
        <div>
          <span style={{ fontSize: "11px", fontWeight: 700, color: "#5bc4f5", letterSpacing: "0.1em" }}>
            WHAT-IF SIMULATOR
          </span>
          <span style={{ fontSize: "10px", color: "#3a5060", marginLeft: "10px" }}>
            Manual state override
          </span>
        </div>
        <button
          onClick={handleReset}
          style={{
            background: "transparent",
            border: "1px solid #2a3a44",
            borderRadius: "4px",
            color: "#3a5060",
            fontSize: "10px",
            padding: "4px 10px",
            cursor: "pointer",
            letterSpacing: "0.06em",
          }}
        >
          RESET
        </button>
      </div>

      <div style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "12px" }}>
        {/* Sliders */}
        <SliderField
          label="CONFUSION"
          value={state.confusion}
          min={0} max={10} step={0.1}
          color={state.confusion > 7 ? "#cf4a2a" : state.confusion > 4 ? "#f5e45b" : "#5bf587"}
          onChange={(v) => set("confusion", v)}
        />
        <SliderField
          label="ATTENTION"
          value={state.attention}
          min={0} max={10} step={0.1}
          color={state.attention < 4 ? "#cf4a2a" : state.attention < 6 ? "#f5e45b" : "#5bc4f5"}
          onChange={(v) => set("attention", v)}
        />
        <SliderField
          label="STEP COUNT"
          value={state.step_count}
          min={0} max={20} step={1}
          color="#e87af5"
          onChange={(v) => set("step_count", v)}
        />
        <SliderField
          label="LAST ACTION IMPACT"
          value={state.last_action_impact}
          min={-1} max={1} step={0.01}
          color={state.last_action_impact >= 0 ? "#5bf587" : "#cf4a2a"}
          onChange={(v) => set("last_action_impact", v)}
        />

        {/* Domain selector */}
        <div>
          <div style={{ fontSize: "9px", color: "#3a5060", letterSpacing: "0.1em", marginBottom: "7px" }}>
            DOMAIN
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "6px" }}>
            {([0, 1, 2, 3] as const).map((idx) => {
              const label = DOMAIN_LABELS[idx];
              const isActive = state.domain_index === idx;
              const colors: Record<HeadType, string> = {
                FACTUAL: "#5bc4f5", PROCEDURAL: "#f5e45b",
                CONCEPTUAL: "#e87af5", TRANSFER: "#5bf587",
              };
              return (
                <button
                  key={idx}
                  onClick={() => set("domain_index", idx)}
                  style={{
                    background: isActive ? `${colors[label]}18` : "#0d1820",
                    border: `1px solid ${isActive ? colors[label] : "#1a2a36"}`,
                    borderRadius: "5px",
                    color: isActive ? colors[label] : "#3a5060",
                    fontSize: "9px",
                    fontWeight: 700,
                    letterSpacing: "0.06em",
                    padding: "7px 4px",
                    cursor: "pointer",
                    fontFamily: "inherit",
                    transition: "all 0.12s ease",
                  }}
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>

        {/* State preview */}
        <div
          style={{
            background: "#0a1218",
            border: "1px solid #1a2a36",
            borderRadius: "5px",
            padding: "10px",
            fontSize: "10px",
            color: "#3a6070",
          }}
        >
          <span style={{ color: "#2a4050" }}>STATE → </span>
          <span style={{ color: "#5a8090" }}>
            {`{confusion: ${state.confusion.toFixed(1)}, attention: ${state.attention.toFixed(1)}, domain: ${state.domain_index}, step: ${state.step_count}, impact: ${state.last_action_impact.toFixed(2)}}`}
          </span>
        </div>

        {/* Run button */}
        <button
          onClick={handleRun}
          disabled={isRunning}
          style={{
            background: isRunning ? "#0d1820" : "#0d2a3a",
            border: `1px solid ${isRunning ? "#1a2a36" : "#1e6fa5"}`,
            borderRadius: "6px",
            color: isRunning ? "#2a4050" : "#5bc4f5",
            fontSize: "11px",
            fontWeight: 700,
            letterSpacing: "0.12em",
            padding: "11px",
            cursor: isRunning ? "not-allowed" : "pointer",
            fontFamily: "inherit",
            transition: "all 0.15s ease",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px",
          }}
        >
          {isRunning ? (
            <>
              <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>◌</span>
              RUNNING DQN FORWARD PASS...
            </>
          ) : (
            "▶ RUN WHAT-IF QUERY"
          )}
          <style>{`@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
        </button>

        {/* Error */}
        {error && (
          <div
            style={{
              padding: "10px",
              background: "#1a0a0a",
              border: "1px solid #cf4a2a44",
              borderRadius: "5px",
              color: "#cf4a2a",
              fontSize: "11px",
            }}
          >
            ⚠ {error}
          </div>
        )}

        {/* Result */}
        {result && <ResultPanel result={result} />}
      </div>
    </div>
  );
}

/* ── Slider ── */
interface SliderFieldProps {
  label: string;
  value: number;
  min: number; max: number; step: number;
  color: string;
  onChange: (v: number) => void;
}

function SliderField({ label, value, min, max, step, color, onChange }: SliderFieldProps) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "9px",
          letterSpacing: "0.1em",
          marginBottom: "5px",
        }}
      >
        <span style={{ color: "#3a5060" }}>{label}</span>
        <span style={{ color, fontWeight: 700 }}>{value.toFixed(step < 1 ? 2 : 0)}</span>
      </div>
      <div style={{ position: "relative", height: "6px" }}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "#1a2a36",
            borderRadius: "3px",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: `${pct}%`,
            background: color,
            borderRadius: "3px",
            opacity: 0.7,
          }}
        />
        <input
          type="range"
          min={min} max={max} step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            opacity: 0,
            cursor: "pointer",
            margin: 0,
          }}
        />
      </div>
    </div>
  );
}

/* ── Result Panel ── */
function ResultPanel({ result }: { result: WhatIfResult }) {
  const actions = Object.entries(result.q_values) as [ActionType, number][];
  const maxQ = Math.max(...actions.map(([, v]) => v));
  const minQ = Math.min(...actions.map(([, v]) => v));
  const range = maxQ - minQ || 1;

  const HEAD_COLORS: Record<HeadType, string> = {
    FACTUAL: "#5bc4f5", PROCEDURAL: "#f5e45b",
    CONCEPTUAL: "#e87af5", TRANSFER: "#5bf587",
  };

  return (
    <div
      style={{
        background: "#0a1218",
        border: "1px solid #1e6fa544",
        borderRadius: "6px",
        overflow: "hidden",
      }}
    >
      {/* Result header */}
      <div
        style={{
          padding: "8px 12px",
          borderBottom: "1px solid #1a2a36",
          display: "flex",
          alignItems: "center",
          gap: "10px",
          background: "#0d1820",
        }}
      >
        <span
          style={{
            fontSize: "9px",
            fontWeight: 700,
            color: HEAD_COLORS[result.selected_head],
            background: `${HEAD_COLORS[result.selected_head]}18`,
            border: `1px solid ${HEAD_COLORS[result.selected_head]}44`,
            padding: "2px 7px",
            borderRadius: "3px",
          }}
        >
          [{result.selected_head}_HEAD]
        </span>
        <span style={{ fontSize: "11px", color: "#9ab8cc", fontWeight: 700 }}>
          → {result.predicted_action.replace("_", " ")}
        </span>
        <span
          style={{
            marginLeft: "auto",
            fontSize: "10px",
            color: result.expected_reward >= 0 ? "#5bf587" : "#cf4a2a",
            fontWeight: 700,
          }}
        >
          {result.expected_reward >= 0 ? "+" : ""}{result.expected_reward.toFixed(1)} r
        </span>
      </div>

      <div style={{ padding: "12px" }}>
        {/* Deltas */}
        <div style={{ display: "flex", gap: "8px", marginBottom: "12px" }}>
          {[
            { label: "Δ CONFUSION", value: result.confusion_delta },
            { label: "Δ ATTENTION",  value: result.attention_delta },
          ].map(({ label, value }) => (
            <div
              key={label}
              style={{
                flex: 1,
                textAlign: "center",
                background: "#080c10",
                border: `1px solid ${value <= 0 ? "#2acf5a33" : "#cf4a2a33"}`,
                borderRadius: "4px",
                padding: "7px",
              }}
            >
              <div
                style={{
                  fontSize: "14px",
                  fontWeight: 700,
                  color: value <= 0 ? "#5bf587" : "#cf4a2a",
                }}
              >
                {value >= 0 ? "+" : ""}{value.toFixed(2)}
              </div>
              <div style={{ fontSize: "9px", color: "#3a5060", marginTop: "2px" }}>{label}</div>
            </div>
          ))}
        </div>

        {/* Q-value bars */}
        <div style={{ fontSize: "9px", color: "#3a5060", letterSpacing: "0.1em", marginBottom: "7px" }}>
          Q-VALUES
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "5px" }}>
          {actions.sort(([, a], [, b]) => b - a).map(([action, q]) => {
            const pct = ((q - minQ) / range) * 100;
            const isBest = action === result.predicted_action;
            return (
              <div key={action} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "9px", color: isBest ? "#9ab8cc" : "#3a5060", width: "110px", flexShrink: 0 }}>
                  {isBest ? "▸ " : "  "}{action.replace("_", " ")}
                </span>
                <div style={{ flex: 1, height: "4px", background: "#1a2a36", borderRadius: "2px" }}>
                  <div
                    style={{
                      height: "100%",
                      width: `${pct}%`,
                      background: isBest ? ACTION_COLORS[action] : "#2a3a44",
                      borderRadius: "2px",
                    }}
                  />
                </div>
                <span style={{ fontSize: "9px", color: isBest ? ACTION_COLORS[action] : "#2a3a44", width: "40px", textAlign: "right" }}>
                  {q.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
