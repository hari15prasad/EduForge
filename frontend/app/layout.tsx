"use client";

import React, { useState, useEffect } from "react";
import { RLProvider, useRL, useAgentStatus, useCurrentDomain, Domain, AgentStatus } from "@/components/RLProvider";

// ─── Types ───────────────────────────────────────────────────────────────────

interface StatBarProps {
  label: string;
  value: number;
  max?: number;
  color: string;
  glowColor: string;
}

interface StatusDotProps {
  status: AgentStatus;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatBar({ label, value, max = 10, color, glowColor }: StatBarProps) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="stat-bar">
      <div className="stat-bar__header">
        <span className="stat-bar__label">{label}</span>
        <span className="stat-bar__value" style={{ color }}>
          {value.toFixed(1)}
        </span>
      </div>
      <div className="stat-bar__track">
        <div
          className="stat-bar__fill"
          style={{
            width: `${pct}%`,
            background: color,
            boxShadow: `0 0 8px ${glowColor}`,
          }}
        />
      </div>
    </div>
  );
}

function StatusDot({ status }: StatusDotProps) {
  const map: Record<AgentStatus, { color: string; label: string }> = {
    online: { color: "#10b981", label: "Online" },
    training: { color: "#f59e0b", label: "Training" },
    evaluating: { color: "#6366f1", label: "Evaluating" },
    idle: { color: "#4a5580", label: "Idle" },
  };
  const { color, label } = map[status];

  return (
    <div className="status-indicator">
      <span
        className="status-indicator__dot"
        style={{
          background: color,
          boxShadow: `0 0 8px ${color}`,
          animation: status !== "idle" ? "pulse 2s infinite" : "none",
        }}
      />
      <span className="status-indicator__label" style={{ color }}>
        {label}
      </span>
    </div>
  );
}

function DomainBadge({ domain }: { domain: Domain }) {
  const colors: Record<Domain, string> = {
    Factual: "#22d3ee",
    Procedural: "#6366f1",
    Conceptual: "#8b5cf6",
    Transfer: "#f59e0b",
  };
  return (
    <span
      className="domain-badge"
      style={{
        borderColor: colors[domain],
        color: colors[domain],
        boxShadow: `0 0 6px ${colors[domain]}44`,
      }}
    >
      {domain}
    </span>
  );
}

function RewardSparkline({ history }: { history: { reward: number }[] }) {
  const last = history.slice(-30);
  if (last.length < 2) {
    return <div className="sparkline sparkline--empty">No data</div>;
  }
  const vals = last.map((e) => e.reward);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const range = max - min || 1;
  const h = 40;
  const w = 220;
  const step = w / (vals.length - 1);

  const pts = vals
    .map((v, i) => `${i * step},${h - ((v - min) / range) * h}`)
    .join(" ");

  return (
    <div className="sparkline">
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h}>
        <defs>
          <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#6366f1" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polyline
          points={pts}
          fill="none"
          stroke="#6366f1"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

// ─── Sidebar ─────────────────────────────────────────────────────────────────

function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const { state } = useRL();

  return (
    <aside className={`sidebar ${collapsed ? "sidebar--collapsed" : ""}`}>
      <button className="sidebar__toggle" onClick={onToggle} aria-label="Toggle sidebar">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path
            d={collapsed ? "M6 3l5 5-5 5" : "M10 3L5 8l5 5"}
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {!collapsed && (
        <>
          <div className="sidebar__section">
            <p className="sidebar__section-title">RL SESSION</p>
            <div className="sidebar__meta">
              <span>Episode</span>
              <span className="sidebar__meta-value">{state.episodeCount}</span>
            </div>
            <div className="sidebar__meta">
              <span>Step</span>
              <span className="sidebar__meta-value">{state.currentStep} / 20</span>
            </div>
            <div className="sidebar__meta">
              <span>ε (Epsilon)</span>
              <span className="sidebar__meta-value">{state.epsilon.toFixed(3)}</span>
            </div>
          </div>

          <div className="sidebar__section">
            <p className="sidebar__section-title">STUDENT STATE</p>
            <StatBar
              label="Confusion"
              value={state.confusion}
              color="#f43f5e"
              glowColor="#f43f5e88"
            />
            <StatBar
              label="Attention"
              value={state.attention}
              color="#10b981"
              glowColor="#10b98188"
            />
          </div>

          <div className="sidebar__section">
            <p className="sidebar__section-title">REWARD HISTORY</p>
            <RewardSparkline history={state.rewardHistory} />
            <div className="sidebar__meta" style={{ marginTop: 8 }}>
              <span>Cumulative</span>
              <span
                className="sidebar__meta-value"
                style={{ color: state.cumulativeReward >= 0 ? "#10b981" : "#f43f5e" }}
              >
                {state.cumulativeReward.toFixed(1)}
              </span>
            </div>
          </div>

          <div className="sidebar__section">
            <p className="sidebar__section-title">DOMAIN PERFORMANCE</p>
            {(["Factual", "Procedural", "Conceptual", "Transfer"] as Domain[]).map((d, i) => {
              const episodes = state.episodeHistory.filter((e) => e.domain === d);
              const successes = episodes.filter((e) => e.outcome === "success").length;
              return (
                <div key={d} className="sidebar__domain-row">
                  <span
                    className="sidebar__domain-dot"
                    style={{ background: ["#22d3ee", "#6366f1", "#8b5cf6", "#f59e0b"][i] }}
                  />
                  <span className="sidebar__domain-name">{d}</span>
                  <span className="sidebar__domain-stat">
                    {successes}/{episodes.length}
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}

      {collapsed && (
        <div className="sidebar__icons">
          {["📊", "🧠", "⚡", "🎯"].map((icon, i) => (
            <span 
              key={i} 
              className="sidebar__icon-btn" 
              title={["Stats", "State", "Rewards", "Domains"][i]}
              onClick={onToggle}
            >
              {icon}
            </span>
          ))}
        </div>
      )}
    </aside>
  );
}

// ─── Header ──────────────────────────────────────────────────────────────────

function Header() {
  const agentStatus = useAgentStatus();
  const currentDomain = useCurrentDomain();
  const { state } = useRL();

  return (
    <header className="header">
      <div className="header__brand">
        <span className="header__logo">⬡</span>
        <span className="header__title">EduForge</span>
        <span className="header__subtitle">DQN Tutor</span>
      </div>

      <div className="header__center">
        <span className="header__domain-label">Active Domain:</span>
        <DomainBadge domain={currentDomain} />
      </div>

      <div className="header__right">
        <div className="header__indicators">
          <StatusDot status="online" />
          <StatusDot status="training" />
          <StatusDot status="evaluating" />
        </div>
        <div className="header__divider" />
        <div className="header__active-status">
          <span className="header__active-label">Agent</span>
          <StatusDot status={agentStatus} />
        </div>
      </div>
    </header>
  );
}

// ─── Root Layout ─────────────────────────────────────────────────────────────

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <RLProvider>
          <Header />
          <div className="app-shell">
            <Sidebar
              collapsed={sidebarCollapsed}
              onToggle={() => setSidebarCollapsed((c) => !c)}
            />
            <main
              className="main-content"
              style={{
                marginLeft: sidebarCollapsed ? "var(--sidebar-collapsed)" : "var(--sidebar-expanded)",
              }}
            >
              {children}
            </main>
          </div>
        </RLProvider>

        <style>{`
          *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

          :root {
            --bg-base: #0a0e1a;
            --bg-surface: #0f1629;
            --bg-elevated: #151e38;
            --bg-overlay: #1a2540;
            --indigo-500: #6366f1;
            --indigo-600: #4f46e5;
            --accent-cyan: #22d3ee;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --text-primary: #e8eeff;
            --text-secondary: #8899cc;
            --text-muted: #4a5580;
            --border-default: #1e2d50;
            --border-accent: #4338ca;
            --sidebar-expanded: 280px;
            --sidebar-collapsed: 64px;
            --header-height: 56px;
            --font-display: 'Space Mono', monospace;
            --font-body: 'DM Sans', system-ui, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
          }

          body {
            background: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-body);
            min-height: 100vh;
            overflow-x: hidden;
          }

          /* Header */
          .header {
            position: fixed;
            top: 0; left: 0; right: 0;
            height: var(--header-height);
            background: rgba(10,14,26,0.92);
            border-bottom: 1px solid var(--border-default);
            backdrop-filter: blur(12px);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            z-index: 100;
            gap: 16px;
          }

          .header__brand {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-shrink: 0;
          }

          .header__logo {
            font-size: 20px;
            color: var(--indigo-500);
            filter: drop-shadow(0 0 6px var(--indigo-500));
          }

          .header__title {
            font-family: var(--font-display);
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: var(--text-primary);
          }

          .header__subtitle {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            letter-spacing: 0.12em;
            text-transform: uppercase;
          }

          .header__center {
            display: flex;
            align-items: center;
            gap: 8px;
          }

          .header__domain-label {
            font-size: 11px;
            color: var(--text-muted);
            font-family: var(--font-mono);
          }

          .header__right {
            display: flex;
            align-items: center;
            gap: 16px;
            flex-shrink: 0;
          }

          .header__indicators {
            display: flex;
            align-items: center;
            gap: 12px;
          }

          .header__divider {
            width: 1px;
            height: 20px;
            background: var(--border-default);
          }

          .header__active-status {
            display: flex;
            align-items: center;
            gap: 8px;
          }

          .header__active-label {
            font-size: 11px;
            color: var(--text-muted);
            font-family: var(--font-mono);
          }

          /* Status indicator */
          .status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
          }

          .status-indicator__dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: block;
          }

          .status-indicator__label {
            font-family: var(--font-mono);
            font-size: 11px;
            letter-spacing: 0.06em;
          }

          /* Domain badge */
          .domain-badge {
            font-family: var(--font-mono);
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid;
            letter-spacing: 0.06em;
          }

          /* App shell */
          .app-shell {
            display: flex;
            padding-top: var(--header-height);
            min-height: 100vh;
          }

          /* Sidebar */
          .sidebar {
            position: fixed;
            top: var(--header-height);
            left: 0;
            bottom: 0;
            width: var(--sidebar-expanded);
            background: var(--bg-surface);
            border-right: 1px solid var(--border-default);
            overflow-y: auto;
            overflow-x: hidden;
            transition: width 220ms cubic-bezier(0.4,0,0.2,1);
            z-index: 90;
            padding: 16px 0;
          }

          .sidebar--collapsed {
            width: var(--sidebar-collapsed);
          }

          .sidebar__toggle {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            margin: 0 auto 16px;
            background: var(--bg-elevated);
            border: 1px solid var(--border-default);
            border-radius: 6px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 120ms ease;
          }

          .sidebar__toggle:hover {
            border-color: var(--indigo-500);
            color: var(--text-primary);
            box-shadow: 0 0 8px rgba(99,102,241,0.3);
          }

          .sidebar__section {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-subtle, #131d38);
          }

          .sidebar__section-title {
            font-family: var(--font-mono);
            font-size: 9px;
            letter-spacing: 0.14em;
            color: var(--text-muted);
            margin-bottom: 10px;
          }

          .sidebar__meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0;
            font-size: 12px;
            color: var(--text-secondary);
          }

          .sidebar__meta-value {
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--text-primary);
          }

          .sidebar__domain-row {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 5px 0;
            font-size: 12px;
          }

          .sidebar__domain-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            flex-shrink: 0;
          }

          .sidebar__domain-name {
            flex: 1;
            color: var(--text-secondary);
          }

          .sidebar__domain-stat {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
          }

          .sidebar__icons {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            padding: 8px 0;
          }

          .sidebar__icon-btn {
            font-size: 18px;
            cursor: pointer;
            padding: 8px;
            border-radius: 6px;
            transition: background 120ms ease;
          }

          .sidebar__icon-btn:hover {
            background: var(--bg-elevated);
          }

          /* Stat bar */
          .stat-bar {
            margin-bottom: 10px;
          }

          .stat-bar__header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
          }

          .stat-bar__label {
            font-size: 11px;
            color: var(--text-secondary);
          }

          .stat-bar__value {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 500;
          }

          .stat-bar__track {
            height: 4px;
            background: var(--bg-elevated);
            border-radius: 2px;
            overflow: hidden;
          }

          .stat-bar__fill {
            height: 100%;
            border-radius: 2px;
            transition: width 300ms ease;
          }

          /* Sparkline */
          .sparkline {
            background: var(--bg-elevated);
            border-radius: 4px;
            padding: 6px;
            border: 1px solid var(--border-default);
          }

          .sparkline--empty {
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            color: var(--text-muted);
          }

          /* Main content */
          .main-content {
            flex: 1;
            transition: margin-left 220ms cubic-bezier(0.4,0,0.2,1);
            padding: 0;
            height: calc(100vh - var(--header-height));
            overflow: hidden;
          }

          /* Animations */
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
          }
        `}</style>
      </body>
    </html>
  );
}
