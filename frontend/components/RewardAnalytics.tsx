"use client";

import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell,
  ResponsiveContainer,
  type TooltipProps,
} from "recharts";
import { useRL, type EpisodeStats } from "@/components/RLProvider";

// ─── Types ───────────────────────────────────────────────────────────────────

interface EpisodeBar {
  episode: number;
  reward: number;
  outcome: EpisodeStats["outcome"];
  domain: string;
  steps: number;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function barColor(reward: number, outcome: EpisodeStats["outcome"]): string {
  if (outcome === "success") return "#10b981";
  if (reward >= 50)  return "#6366f1";
  if (reward >= 0)   return "#f59e0b";
  return "#f43f5e";
}

function barGlow(color: string): string {
  return `drop-shadow(0 0 4px ${color}88)`;
}

// ─── Custom Tooltip ───────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as EpisodeBar;
  const color = barColor(d.reward, d.outcome);

  return (
    <div className="ra-tooltip">
      <span className="ra-tooltip__ep">Episode {label}</span>
      <div className="ra-tooltip__row">
        <span className="ra-tooltip__key">Reward</span>
        <span className="ra-tooltip__val" style={{ color }}>
          {d.reward > 0 ? "+" : ""}{d.reward.toFixed(1)}
        </span>
      </div>
      <div className="ra-tooltip__row">
        <span className="ra-tooltip__key">Outcome</span>
        <span
          className="ra-tooltip__val"
          style={{ color: d.outcome === "success" ? "#10b981" : d.outcome === "timeout" ? "#f43f5e" : "#8899cc" }}
        >
          {d.outcome.toUpperCase()}
        </span>
      </div>
      <div className="ra-tooltip__row">
        <span className="ra-tooltip__key">Domain</span>
        <span className="ra-tooltip__val" style={{ color: "#818cf8" }}>{d.domain}</span>
      </div>
      <div className="ra-tooltip__row">
        <span className="ra-tooltip__key">Steps</span>
        <span className="ra-tooltip__val">{d.steps}</span>
      </div>
      <style>{`
        .ra-tooltip {
          background: rgba(10,14,26,0.96);
          border: 1px solid #1e2d50;
          border-radius: 8px;
          padding: 10px 12px;
          display: flex;
          flex-direction: column;
          gap: 4px;
          font-family: 'JetBrains Mono', monospace;
          box-shadow: 0 4px 24px rgba(0,0,0,0.5);
          min-width: 160px;
        }
        .ra-tooltip__ep { font-size: 9px; color: #4a5580; letter-spacing: 0.1em; margin-bottom: 2px; }
        .ra-tooltip__row { display: flex; justify-content: space-between; gap: 16px; }
        .ra-tooltip__key { font-size: 11px; color: #8899cc; }
        .ra-tooltip__val { font-size: 11px; font-weight: 700; color: #e8eeff; }
      `}</style>
    </div>
  );
}

// ─── Custom Bar Shape ─────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function RoundBar(props: any) {
  const { x, y, width, height, fill } = props;
  if (!height || height === 0) return null;
  const isNeg = height < 0;
  const absH = Math.abs(height);
  const r = Math.min(4, width / 2);

  return (
    <g style={{ filter: barGlow(fill) }}>
      <path
        d={
          isNeg
            ? `M${x},${y} L${x + width},${y} L${x + width},${y + absH - r} Q${x + width},${y + absH} ${x + width - r},${y + absH} L${x + r},${y + absH} Q${x},${y + absH} ${x},${y + absH - r} Z`
            : `M${x + r},${y} Q${x},${y} ${x},${y + r} L${x},${y + absH} L${x + width},${y + absH} L${x + width},${y + r} Q${x + width},${y} ${x + width - r},${y} Z`
        }
        fill={fill}
        opacity={0.9}
      />
    </g>
  );
}

// ─── Summary Stats ────────────────────────────────────────────────────────────

interface SummaryStatProps { label: string; value: string | number; color?: string }
function SummaryStat({ label, value, color = "#e8eeff" }: SummaryStatProps) {
  return (
    <div className="ra-stat">
      <span className="ra-stat__label">{label}</span>
      <span className="ra-stat__value" style={{ color }}>{value}</span>
      <style>{`
        .ra-stat { display: flex; flex-direction: column; align-items: center; gap: 2px; padding: 8px 12px; background: rgba(15,22,41,0.8); border: 1px solid #1e2d50; border-radius: 10px; }
        .ra-stat__label { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #4a5580; letter-spacing: 0.1em; }
        .ra-stat__value { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 700; }
      `}</style>
    </div>
  );
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function RewardAnalytics() {
  const { state } = useRL();

  const chartData: EpisodeBar[] = useMemo(() => {
    if (!state.episodeHistory.length) return [];
    return state.episodeHistory.slice(-30).map((ep, i) => ({
      episode: ep.episodeId ?? i + 1,
      reward: ep.totalReward,
      outcome: ep.outcome,
      domain: ep.domain,
      steps: ep.steps,
    }));
  }, [state.episodeHistory]);

  // Summary stats
  const totalEps = state.episodeHistory.length;
  const successes = state.episodeHistory.filter((e) => e.outcome === "success").length;
  const successRate = totalEps ? ((successes / totalEps) * 100).toFixed(0) : "—";
  const avgReward = totalEps
    ? (state.episodeHistory.reduce((s, e) => s + e.totalReward, 0) / totalEps).toFixed(1)
    : "—";
  const bestReward = totalEps
    ? Math.max(...state.episodeHistory.map((e) => e.totalReward)).toFixed(1)
    : "—";

  const minY = chartData.length ? Math.min(...chartData.map((d) => d.reward), -10) - 5 : -10;
  const maxY = chartData.length ? Math.max(...chartData.map((d) => d.reward), 10) + 5  : 110;

  return (
    <section className="ra-section">
      <div className="ra-header">
        <div>
          <span className="ra-title">REWARD ANALYTICS</span>
          <span className="ra-subtitle">Per-episode cumulative reward · last 30 episodes</span>
        </div>
        <div className="ra-summary">
          <SummaryStat label="EPISODES" value={totalEps} />
          <SummaryStat label="SUCCESS %" value={`${successRate}%`} color="#10b981" />
          <SummaryStat label="AVG REWARD" value={avgReward} color="#6366f1" />
          <SummaryStat label="BEST" value={bestReward} color="#f59e0b" />
        </div>
      </div>

      {chartData.length === 0 ? (
        <div className="ra-empty">
          <span>⬡</span>
          <span>No episodes recorded yet</span>
          <span>Complete an episode to populate analytics</span>
        </div>
      ) : (
        <div className="ra-chart">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} margin={{ top: 10, right: 8, left: -20, bottom: 0 }} barCategoryGap="20%">
              <CartesianGrid stroke="#1a2540" strokeDasharray="3 3" vertical={false} />

              <XAxis
                dataKey="episode"
                tick={{ fill: "#4a5580", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
                tickLine={false}
                axisLine={{ stroke: "#1e2d50" }}
                label={{ value: "Episode", position: "insideBottomRight", offset: -4, fill: "#4a5580", fontSize: 10 }}
              />

              <YAxis
                domain={[minY, maxY]}
                tick={{ fill: "#4a5580", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
                tickLine={false}
                axisLine={false}
                width={30}
              />

              <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(99,102,241,0.06)" }} />

              {/* Zero line */}
              <ReferenceLine y={0} stroke="#1e2d50" strokeWidth={1} />

              {/* Success reward line (approx +100 - steps*1) */}
              <ReferenceLine
                y={80}
                stroke="#10b981"
                strokeDasharray="4 4"
                strokeOpacity={0.4}
                label={{ value: "Good", fill: "#10b981", fontSize: 9, position: "right", fontFamily: "'JetBrains Mono', monospace" }}
              />

              <Bar dataKey="reward" shape={<RoundBar />} isAnimationActive animationDuration={500} maxBarSize={32}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={barColor(entry.reward, entry.outcome)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Domain breakdown */}
      {totalEps > 0 && (
        <div className="ra-domains">
          {(["Factual", "Procedural", "Conceptual", "Transfer"] as const).map((domain, i) => {
            const eps = state.episodeHistory.filter((e) => e.domain === domain);
            const rate = eps.length
              ? ((eps.filter((e) => e.outcome === "success").length / eps.length) * 100).toFixed(0)
              : 0;
            const colors = ["#22d3ee", "#6366f1", "#8b5cf6", "#f59e0b"];
            return (
              <div key={domain} className="ra-domain-chip">
                <span className="ra-domain-dot" style={{ background: colors[i] }} />
                <span className="ra-domain-name">{domain}</span>
                <span className="ra-domain-rate" style={{ color: colors[i] }}>{rate}%</span>
              </div>
            );
          })}
        </div>
      )}

      <style>{`
        .ra-section {
          background: rgba(10,14,26,0.6);
          border: 1px solid #1e2d50;
          border-radius: 20px;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 14px;
        }
        .ra-header { display: flex; flex-direction: column; gap: 12px; }
        .ra-title {
          display: block;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          letter-spacing: 0.16em;
          color: #4a5580;
        }
        .ra-subtitle {
          display: block;
          font-family: 'DM Sans', system-ui, sans-serif;
          font-size: 12px;
          color: #8899cc;
          margin-top: 2px;
        }
        .ra-summary { display: flex; gap: 8px; flex-wrap: wrap; }
        .ra-chart { margin: 0 -4px; }
        .ra-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 180px;
          gap: 6px;
          color: #4a5580;
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
        }
        .ra-empty span:first-child { font-size: 28px; opacity: 0.4; }
        .ra-empty span:nth-child(2) { color: #8899cc; font-size: 13px; }
        .ra-domains { display: flex; gap: 8px; flex-wrap: wrap; }
        .ra-domain-chip {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 5px 10px;
          background: rgba(15,22,41,0.8);
          border: 1px solid #1e2d50;
          border-radius: 20px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
        }
        .ra-domain-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
        .ra-domain-name { color: #8899cc; }
        .ra-domain-rate { font-weight: 700; }
      `}</style>
    </section>
  );
}
