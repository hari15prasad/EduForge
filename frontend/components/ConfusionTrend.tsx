"use client";

import React, { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  type TooltipProps,
} from "recharts";
import { useRL } from "@/components/RLProvider";

// ─── Types ───────────────────────────────────────────────────────────────────

interface ChartPoint {
  step: number;
  confusion: number;
  delta: number;
  attention: number;
}

// ─── Custom Tooltip ──────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as ChartPoint;

  return (
    <div className="ct-tooltip">
      <span className="ct-tooltip__step">Step {label}</span>
      <div className="ct-tooltip__row">
        <span className="ct-tooltip__key">Confusion</span>
        <span className="ct-tooltip__val" style={{ color: confusionColor(d.confusion) }}>
          {d.confusion.toFixed(2)}
        </span>
      </div>
      <div className="ct-tooltip__row">
        <span className="ct-tooltip__key">Δ Delta</span>
        <span
          className="ct-tooltip__val"
          style={{ color: d.delta < 0 ? "#10b981" : d.delta > 0 ? "#f43f5e" : "#8899cc" }}
        >
          {d.delta > 0 ? "+" : ""}{d.delta.toFixed(2)}
        </span>
      </div>
      <div className="ct-tooltip__row">
        <span className="ct-tooltip__key">Attention</span>
        <span className="ct-tooltip__val" style={{ color: "#818cf8" }}>{d.attention.toFixed(2)}</span>
      </div>
      <style>{`
        .ct-tooltip {
          background: rgba(10,14,26,0.95);
          border: 1px solid #1e2d50;
          border-radius: 8px;
          padding: 10px 12px;
          display: flex;
          flex-direction: column;
          gap: 4px;
          font-family: 'JetBrains Mono', monospace;
          box-shadow: 0 4px 24px rgba(0,0,0,0.5);
        }
        .ct-tooltip__step {
          font-size: 9px;
          color: #4a5580;
          letter-spacing: 0.1em;
          margin-bottom: 2px;
        }
        .ct-tooltip__row { display: flex; justify-content: space-between; gap: 16px; }
        .ct-tooltip__key { font-size: 11px; color: #8899cc; }
        .ct-tooltip__val { font-size: 11px; font-weight: 700; }
      `}</style>
    </div>
  );
}

function confusionColor(v: number): string {
  if (v <= 3) return "#10b981";
  if (v <= 6) return "#f59e0b";
  return "#f43f5e";
}

// ─── Dot ─────────────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ActiveDot(props: any) {
  const { cx, cy, payload } = props;
  const color = confusionColor(payload.confusion);
  return (
    <circle
      cx={cx} cy={cy} r={5}
      fill={color}
      stroke="rgba(10,14,26,0.8)"
      strokeWidth={2}
      style={{ filter: `drop-shadow(0 0 4px ${color})` }}
    />
  );
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ConfusionTrend() {
  const { state } = useRL();

  // Build chart data from rewardHistory (step-indexed) + current confusion
  const data: ChartPoint[] = useMemo(() => {
    const history = state.rewardHistory;
    if (!history.length) {
      // Seed with initial state
      return [{ step: 0, confusion: state.confusion, delta: 0, attention: state.attention }];
    }

    const points: ChartPoint[] = [];
    let prevConfusion = state.confusion + history.reduce((s, r) => s - r.reward * 0.08, 0);
    prevConfusion = Math.min(10, Math.max(0, prevConfusion));

    history.forEach((entry, i) => {
      // Simulate confusion level at each step via reward signal
      const confusion = Math.min(10, Math.max(0, prevConfusion - entry.reward * 0.06));
      const delta = i === 0 ? 0 : confusion - points[i - 1].confusion;
      points.push({ step: entry.step, confusion: +confusion.toFixed(2), delta: +delta.toFixed(2), attention: state.attention });
      prevConfusion = confusion;
    });

    return points;
  }, [state.rewardHistory, state.confusion, state.attention]);

  const minC = Math.min(...data.map((d) => d.confusion), 0);
  const maxC = Math.max(...data.map((d) => d.confusion), 10);

  return (
    <section className="ct-section">
      <div className="ct-header">
        <div>
          <span className="ct-title">CONFUSION TRAJECTORY</span>
          <span className="ct-subtitle">Δ confusion across episode steps</span>
        </div>
        <div className="ct-badges">
          <span className="ct-badge ct-badge--success">✓ Goal ≤ 2.0</span>
          <span className="ct-badge ct-badge--current">
            Now: {state.confusion.toFixed(1)}
          </span>
        </div>
      </div>

      <div className="ct-chart">
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={data} margin={{ top: 10, right: 8, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="confusionGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#f43f5e" stopOpacity={0.35} />
                <stop offset="60%" stopColor="#f59e0b" stopOpacity={0.1} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="attentionGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <CartesianGrid
              stroke="#1a2540"
              strokeDasharray="3 3"
              vertical={false}
            />

            <XAxis
              dataKey="step"
              tick={{ fill: "#4a5580", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
              tickLine={false}
              axisLine={{ stroke: "#1e2d50" }}
              label={{ value: "Step", position: "insideBottomRight", offset: -4, fill: "#4a5580", fontSize: 10 }}
            />

            <YAxis
              domain={[Math.max(0, minC - 0.5), Math.min(10, maxC + 0.5)]}
              tick={{ fill: "#4a5580", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
              tickLine={false}
              axisLine={false}
              width={30}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Success threshold line */}
            <ReferenceLine
              y={2}
              stroke="#10b981"
              strokeDasharray="4 4"
              strokeOpacity={0.6}
              label={{ value: "✓ 2.0", fill: "#10b981", fontSize: 10, position: "right", fontFamily: "'JetBrains Mono', monospace" }}
            />

            {/* Danger threshold line */}
            <ReferenceLine
              y={7}
              stroke="#f43f5e"
              strokeDasharray="4 4"
              strokeOpacity={0.4}
              label={{ value: "⚠ 7.0", fill: "#f43f5e", fontSize: 10, position: "right", fontFamily: "'JetBrains Mono', monospace" }}
            />

            <Area
              type="monotone"
              dataKey="attention"
              stroke="#6366f1"
              strokeWidth={1}
              strokeDasharray="3 3"
              fill="url(#attentionGrad)"
              dot={false}
              strokeOpacity={0.5}
              isAnimationActive={true}
              animationDuration={600}
            />

            <Area
              type="monotone"
              dataKey="confusion"
              stroke="#f43f5e"
              strokeWidth={2}
              fill="url(#confusionGrad)"
              dot={false}
              activeDot={<ActiveDot />}
              isAnimationActive={true}
              animationDuration={600}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="ct-legend">
        <span className="ct-legend-item">
          <span className="ct-legend-line" style={{ background: "#f43f5e" }} />
          Confusion
        </span>
        <span className="ct-legend-item">
          <span className="ct-legend-line" style={{ background: "#6366f1", opacity: 0.5 }} />
          Attention (ref)
        </span>
        <span className="ct-legend-item">
          <span className="ct-legend-line" style={{ background: "#10b981", borderTop: "2px dashed #10b981", width: 20 }} />
          Success threshold
        </span>
      </div>

      <style>{`
        .ct-section {
          background: rgba(10,14,26,0.6);
          border: 1px solid #1e2d50;
          border-radius: 20px;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .ct-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 8px;
        }
        .ct-title {
          display: block;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          letter-spacing: 0.16em;
          color: #4a5580;
        }
        .ct-subtitle {
          display: block;
          font-family: 'DM Sans', system-ui, sans-serif;
          font-size: 12px;
          color: #8899cc;
          margin-top: 2px;
        }
        .ct-badges { display: flex; gap: 6px; align-items: center; }
        .ct-badge {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          padding: 3px 8px;
          border-radius: 5px;
          border: 1px solid;
        }
        .ct-badge--success { color: #10b981; border-color: rgba(16,185,129,0.4); background: rgba(16,185,129,0.08); }
        .ct-badge--current { color: #f59e0b; border-color: rgba(245,158,11,0.4); background: rgba(245,158,11,0.08); }
        .ct-chart { margin: 0 -4px; }
        .ct-legend {
          display: flex;
          gap: 16px;
          flex-wrap: wrap;
          justify-content: center;
        }
        .ct-legend-item {
          display: flex;
          align-items: center;
          gap: 6px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          color: #8899cc;
        }
        .ct-legend-line {
          width: 20px;
          height: 2px;
          border-radius: 1px;
          display: block;
        }
      `}</style>
    </section>
  );
}
