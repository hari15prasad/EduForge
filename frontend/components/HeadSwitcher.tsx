import { useState } from "react";

export type HeadType = "FACTUAL" | "PROCEDURAL" | "CONCEPTUAL" | "TRANSFER";

export interface HeadStats {
  head: HeadType;
  episodeRewards: number[];
  successRate: number;   // 0–1
  avgSteps: number;
  totalEpisodes: number;
}

interface HeadSwitcherProps {
  stats: HeadStats[];
  activeHead: HeadType;
  onHeadChange: (head: HeadType) => void;
}

const HEAD_META: Record<HeadType, { icon: string; color: string; glow: string; desc: string }> = {
  FACTUAL:    { icon: "◈", color: "#5bc4f5", glow: "#1e6fa5", desc: "Factual recall & definitions" },
  PROCEDURAL: { icon: "◎", color: "#f5e45b", glow: "#c4a81e", desc: "Step-by-step procedures" },
  CONCEPTUAL: { icon: "◬", color: "#e87af5", glow: "#b42ecf", desc: "Concept & principle mastery" },
  TRANSFER:   { icon: "◉", color: "#5bf587", glow: "#2acf5a", desc: "Cross-domain application" },
};

const HEADS: HeadType[] = ["FACTUAL", "PROCEDURAL", "CONCEPTUAL", "TRANSFER"];

export default function HeadSwitcher({ stats, activeHead, onHeadChange }: HeadSwitcherProps) {
  const activeStat = stats.find((s) => s.head === activeHead);

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
      {/* Tab bar */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          borderBottom: "1px solid #1a2a36",
        }}
      >
        {HEADS.map((head) => {
          const meta = HEAD_META[head];
          const stat = stats.find((s) => s.head === head);
          const isActive = head === activeHead;

          return (
            <button
              key={head}
              onClick={() => onHeadChange(head)}
              style={{
                background: isActive ? `${meta.glow}18` : "transparent",
                border: "none",
                borderBottom: isActive ? `2px solid ${meta.color}` : "2px solid transparent",
                padding: "12px 8px 10px",
                cursor: "pointer",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "4px",
                transition: "all 0.15s ease",
                position: "relative",
              }}
            >
              <span
                style={{
                  fontSize: "18px",
                  color: isActive ? meta.color : "#3a5060",
                  textShadow: isActive ? `0 0 12px ${meta.glow}` : "none",
                  transition: "all 0.15s ease",
                }}
              >
                {meta.icon}
              </span>
              <span
                style={{
                  fontSize: "9px",
                  fontWeight: 700,
                  letterSpacing: "0.1em",
                  color: isActive ? meta.color : "#3a5060",
                  transition: "color 0.15s ease",
                }}
              >
                {head}
              </span>
              {stat && (
                <span
                  style={{
                    fontSize: "9px",
                    color: isActive ? "#7aaabb" : "#2a3a44",
                  }}
                >
                  {(stat.successRate * 100).toFixed(0)}% ✓
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Panel */}
      {activeStat ? (
        <HeadPanel stat={activeStat} />
      ) : (
        <div style={{ padding: "24px", color: "#3a5060", fontSize: "12px", textAlign: "center" }}>
          No data for {activeHead}
        </div>
      )}
    </div>
  );
}

function HeadPanel({ stat }: { stat: HeadStats }) {
  const meta = HEAD_META[stat.head];
  const recent = stat.episodeRewards.slice(-30);
  const maxR = Math.max(...recent, 1);
  const minR = Math.min(...recent, 0);
  const range = maxR - minR || 1;

  return (
    <div style={{ padding: "16px" }}>
      {/* Description */}
      <div
        style={{
          fontSize: "11px",
          color: "#5a7a8a",
          marginBottom: "16px",
          borderLeft: `2px solid ${meta.glow}`,
          paddingLeft: "10px",
        }}
      >
        {meta.desc}
      </div>

      {/* Stats row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "10px",
          marginBottom: "16px",
        }}
      >
        {[
          { label: "Success Rate", value: `${(stat.successRate * 100).toFixed(1)}%` },
          { label: "Avg Steps",    value: stat.avgSteps.toFixed(1) },
          { label: "Episodes",     value: stat.totalEpisodes.toLocaleString() },
        ].map(({ label, value }) => (
          <div
            key={label}
            style={{
              background: "#0d1820",
              border: `1px solid ${meta.glow}33`,
              borderRadius: "6px",
              padding: "10px",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: "16px", fontWeight: 700, color: meta.color }}>{value}</div>
            <div style={{ fontSize: "9px", color: "#3a5060", marginTop: "3px", letterSpacing: "0.08em" }}>
              {label.toUpperCase()}
            </div>
          </div>
        ))}
      </div>

      {/* Reward sparkline */}
      <div>
        <div
          style={{
            fontSize: "9px",
            color: "#3a5060",
            letterSpacing: "0.1em",
            marginBottom: "8px",
          }}
        >
          RECENT EPISODE REWARDS (last {recent.length})
        </div>
        <div
          style={{
            position: "relative",
            height: "56px",
            background: "#0a1218",
            border: `1px solid ${meta.glow}22`,
            borderRadius: "4px",
            padding: "4px",
            display: "flex",
            alignItems: "flex-end",
            gap: "1px",
          }}
        >
          {recent.map((r, i) => {
            const h = ((r - minR) / range) * 44 + 4;
            const isPositive = r >= 0;
            return (
              <div
                key={i}
                title={`Ep ${stat.totalEpisodes - recent.length + i}: ${r.toFixed(1)}`}
                style={{
                  flex: 1,
                  height: `${h}px`,
                  background: isPositive ? meta.color : "#cf4a2a",
                  borderRadius: "1px",
                  opacity: 0.7 + (i / recent.length) * 0.3,
                  transition: "height 0.2s ease",
                  cursor: "default",
                }}
              />
            );
          })}
          {/* Zero line */}
          {minR < 0 && (
            <div
              style={{
                position: "absolute",
                left: 4,
                right: 4,
                bottom: `${((-minR) / range) * 44 + 4}px`,
                height: "1px",
                background: "#2a3a44",
              }}
            />
          )}
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "9px",
            color: "#2a3a44",
            marginTop: "4px",
          }}
        >
          <span>{minR.toFixed(0)}</span>
          <span>{maxR.toFixed(0)}</span>
        </div>
      </div>
    </div>
  );
}
