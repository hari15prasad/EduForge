"use client";

import React, { useEffect, useRef, useState } from "react";
import { useConfusion, useAttention } from "@/components/RLProvider";

// ─── Types ───────────────────────────────────────────────────────────────────

interface GaugeConfig {
  value: number;       // 0–10
  label: string;
  sublabel: string;
  invert?: boolean;    // true = high value is bad (Confusion)
}

interface CircularGaugeProps extends GaugeConfig {
  size?: number;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function getPct(value: number, invert = false): number {
  const raw = Math.min(100, Math.max(0, (value / 10) * 100));
  return invert ? 100 - raw : raw;
}

type ColorTier = "green" | "yellow" | "red";

function getTier(pct: number): ColorTier {
  if (pct >= 70) return "green";
  if (pct >= 40) return "yellow";
  return "red";
}

const TIER_COLORS: Record<ColorTier, { stroke: string; glow: string; text: string; bg: string }> = {
  green:  { stroke: "#10b981", glow: "rgba(16,185,129,0.55)",  text: "#34d399", bg: "rgba(16,185,129,0.08)" },
  yellow: { stroke: "#f59e0b", glow: "rgba(245,158,11,0.55)",  text: "#fbbf24", bg: "rgba(245,158,11,0.08)" },
  red:    { stroke: "#f43f5e", glow: "rgba(244,63,94,0.55)",   text: "#fb7185", bg: "rgba(244,63,94,0.08)" },
};

// ─── Animated Number ─────────────────────────────────────────────────────────

function AnimatedNumber({ value, decimals = 1 }: { value: number; decimals?: number }) {
  const [display, setDisplay] = useState(value);
  const raf = useRef<number | null>(null);
  const start = useRef<number | null>(null);
  const from = useRef(value);

  useEffect(() => {
    from.current = display;
    start.current = null;
    if (raf.current) cancelAnimationFrame(raf.current);

    const animate = (ts: number) => {
      if (!start.current) start.current = ts;
      const progress = Math.min((ts - start.current) / 350, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(from.current + (value - from.current) * eased);
      if (progress < 1) raf.current = requestAnimationFrame(animate);
    };

    raf.current = requestAnimationFrame(animate);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return <>{display.toFixed(decimals)}</>;
}

// ─── Circular Gauge ───────────────────────────────────────────────────────────

function CircularGauge({ value, label, sublabel, invert = false, size = 180 }: CircularGaugeProps) {
  const pct = getPct(value, invert);
  const tier = getTier(pct);
  const colors = TIER_COLORS[tier];

  const cx = size / 2;
  const cy = size / 2;
  const strokeW = 8;
  const r = (size - strokeW * 2 - 8) / 2;
  const circumference = 2 * Math.PI * r;

  // Arc: start from top (-90deg), sweep clockwise
  const dashOffset = circumference * (1 - pct / 100);

  // Tick marks (20 ticks around the ring)
  const ticks = Array.from({ length: 20 }, (_, i) => {
    const angle = (i / 20) * 360 - 90;
    const rad = (angle * Math.PI) / 180;
    const inner = r + strokeW / 2 + 4;
    const outer = inner + (i % 5 === 0 ? 8 : 4);
    return {
      x1: cx + inner * Math.cos(rad),
      y1: cy + inner * Math.sin(rad),
      x2: cx + outer * Math.cos(rad),
      y2: cy + outer * Math.sin(rad),
      major: i % 5 === 0,
    };
  });

  return (
    <div className="gauge-wrap" style={{ "--glow": colors.glow, "--stroke": colors.stroke } as React.CSSProperties}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="gauge-svg"
        role="img"
        aria-label={`${label}: ${value.toFixed(1)}`}
      >
        <defs>
          <filter id={`blur-${label}`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
          </filter>
          <linearGradient id={`grad-${label}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor={colors.stroke} stopOpacity="1" />
            <stop offset="100%" stopColor={colors.stroke} stopOpacity="0.6" />
          </linearGradient>
        </defs>

        {/* Background ring */}
        <circle
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke="#1a2540"
          strokeWidth={strokeW}
        />

        {/* Glow ring (duplicate, blurred) */}
        <circle
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke={colors.stroke}
          strokeWidth={strokeW + 4}
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          transform={`rotate(-90 ${cx} ${cy})`}
          opacity="0.25"
          filter={`url(#blur-${label})`}
          style={{ transition: "stroke-dashoffset 600ms cubic-bezier(0.4,0,0.2,1), stroke 400ms ease" }}
        />

        {/* Main progress arc */}
        <circle
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke={`url(#grad-${label})`}
          strokeWidth={strokeW}
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{ transition: "stroke-dashoffset 600ms cubic-bezier(0.4,0,0.2,1), stroke 400ms ease" }}
        />

        {/* Tick marks */}
        {ticks.map((t, i) => (
          <line
            key={i}
            x1={t.x1} y1={t.y1}
            x2={t.x2} y2={t.y2}
            stroke={t.major ? "#2d3d5a" : "#1e2d50"}
            strokeWidth={t.major ? 1.5 : 1}
          />
        ))}

        {/* Center content */}
        <foreignObject x={cx - 52} y={cy - 36} width={104} height={72}>
          <div className="gauge-center">
            <span className="gauge-value" style={{ color: colors.text }}>
              <AnimatedNumber value={value} />
            </span>
            <span className="gauge-max">/10</span>
            <span className="gauge-tier" style={{ color: colors.text }}>
              {tier.toUpperCase()}
            </span>
          </div>
        </foreignObject>
      </svg>

      <div className="gauge-label">{label}</div>
      <div className="gauge-sublabel">{sublabel}</div>

      <style>{`
        .gauge-wrap {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 6px;
          position: relative;
          padding: 12px;
          background: rgba(15,22,41,0.7);
          border: 1px solid rgba(30,45,80,0.8);
          border-radius: 16px;
          transition: border-color 400ms ease, box-shadow 400ms ease;
        }
        .gauge-wrap:hover {
          border-color: var(--stroke);
          box-shadow: 0 0 24px var(--glow), 0 4px 32px rgba(0,0,0,0.4);
        }
        .gauge-svg {
          filter: drop-shadow(0 0 6px var(--glow));
          transition: filter 400ms ease;
        }
        .gauge-center {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          font-family: 'JetBrains Mono', 'Space Mono', monospace;
        }
        .gauge-value {
          font-size: 26px;
          font-weight: 700;
          line-height: 1;
          letter-spacing: -0.02em;
          transition: color 400ms ease;
        }
        .gauge-max {
          font-size: 11px;
          color: #4a5580;
          margin-top: 1px;
        }
        .gauge-tier {
          font-size: 9px;
          letter-spacing: 0.12em;
          margin-top: 4px;
          transition: color 400ms ease;
        }
        .gauge-label {
          font-family: 'DM Sans', system-ui, sans-serif;
          font-size: 13px;
          font-weight: 600;
          color: #e8eeff;
          letter-spacing: 0.02em;
        }
        .gauge-sublabel {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          color: #4a5580;
          letter-spacing: 0.08em;
        }
      `}</style>
    </div>
  );
}

// ─── State Gauges ─────────────────────────────────────────────────────────────

export default function StateGauges() {
  const confusion = useConfusion();
  const attention = useAttention();

  return (
    <section className="gauges-section">
      <header className="gauges-header">
        <span className="gauges-title">STUDENT STATE</span>
        <span className="gauges-badge">LIVE</span>
      </header>

      <div className="gauges-grid">
        <CircularGauge
          value={attention}
          label="Student Attention"
          sublabel="ATT · 0–10"
          invert={false}
        />
        <CircularGauge
          value={confusion}
          label="Student Confusion"
          sublabel="CNF · 0–10"
          invert={true}
        />
      </div>

      <div className="gauges-legend">
        {(["green", "yellow", "red"] as ColorTier[]).map((t) => (
          <span key={t} className="legend-item">
            <span className="legend-dot" style={{ background: TIER_COLORS[t].stroke }} />
            <span className="legend-text" style={{ color: TIER_COLORS[t].text }}>
              {t === "green" ? "≥70%" : t === "yellow" ? "40–70%" : "<40%"}
            </span>
          </span>
        ))}
      </div>

      <style>{`
        .gauges-section {
          background: rgba(10,14,26,0.6);
          border: 1px solid #1e2d50;
          border-radius: 20px;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .gauges-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }
        .gauges-title {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          letter-spacing: 0.16em;
          color: #4a5580;
        }
        .gauges-badge {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          letter-spacing: 0.12em;
          padding: 2px 7px;
          border-radius: 4px;
          background: rgba(16,185,129,0.12);
          color: #10b981;
          border: 1px solid rgba(16,185,129,0.3);
          animation: livePulse 2s ease-in-out infinite;
        }
        @keyframes livePulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        .gauges-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }
        .gauges-legend {
          display: flex;
          justify-content: center;
          gap: 16px;
        }
        .legend-item {
          display: flex;
          align-items: center;
          gap: 5px;
        }
        .legend-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
        }
        .legend-text {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
        }
      `}</style>
    </section>
  );
}
