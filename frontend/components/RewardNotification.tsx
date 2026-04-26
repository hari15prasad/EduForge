"use client";

import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  createContext,
  useContext,
  type ReactNode,
} from "react";
import { useRL, type Action } from "@/components/RLProvider";

// ─── Types ───────────────────────────────────────────────────────────────────

interface RewardToast {
  id: string;
  reward: number;
  action: Action;
  step: number;
  timestamp: number;
}

interface ToastContextValue {
  toasts: RewardToast[];
  push: (toast: Omit<RewardToast, "id">) => void;
  dismiss: (id: string) => void;
}

// ─── Context ─────────────────────────────────────────────────────────────────

const ToastContext = createContext<ToastContextValue | null>(null);

export function RewardToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<RewardToast[]>([]);

  const push = useCallback((toast: Omit<RewardToast, "id">) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    setToasts((prev) => [...prev.slice(-4), { ...toast, id }]); // cap at 5
  }, []);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toasts, push, dismiss }}>
      {children}
      <RewardNotificationStack />
    </ToastContext.Provider>
  );
}

function useToastContext(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToastContext must be used inside <RewardToastProvider>");
  return ctx;
}

// ─── Hook: useRewardNotifier ──────────────────────────────────────────────────
// Watches rewardHistory and fires a toast on each new entry.

export function useRewardNotifier() {
  const { state } = useRL();
  const { push } = useToastContext();
  const lastLenRef = useRef(state.rewardHistory.length);

  useEffect(() => {
    const current = state.rewardHistory.length;
    if (current > lastLenRef.current) {
      const newEntries = state.rewardHistory.slice(lastLenRef.current);
      newEntries.forEach((entry) => {
        push({
          reward: entry.reward,
          action: entry.action,
          step: entry.step,
          timestamp: entry.timestamp,
        });
      });
      lastLenRef.current = current;
    }
  }, [state.rewardHistory, push]);
}

// ─── Action Labels ────────────────────────────────────────────────────────────

const ACTION_META: Record<Action, { short: string; icon: string }> = {
  explain:        { short: "EXP", icon: "◈" },
  correct_fact:   { short: "COR", icon: "⊕" },
  worked_example: { short: "WEX", icon: "▦" },
  analogize:      { short: "ANA", icon: "⟷" },
  question:       { short: "QST", icon: "⊙" },
};

// ─── Toast Color ──────────────────────────────────────────────────────────────

function toastPalette(reward: number): {
  border: string;
  glow: string;
  accent: string;
  bg: string;
  label: string;
} {
  if (reward >= 50)
    return { border: "#10b981", glow: "rgba(16,185,129,0.5)",  accent: "#34d399", bg: "rgba(16,185,129,0.08)", label: "CLEARED" };
  if (reward >= 5)
    return { border: "#6366f1", glow: "rgba(99,102,241,0.5)",  accent: "#818cf8", bg: "rgba(99,102,241,0.08)",  label: "POSITIVE" };
  if (reward >= 0)
    return { border: "#f59e0b", glow: "rgba(245,158,11,0.4)",  accent: "#fbbf24", bg: "rgba(245,158,11,0.06)", label: "NEUTRAL" };
  return   { border: "#f43f5e", glow: "rgba(244,63,94,0.5)",   accent: "#fb7185", bg: "rgba(244,63,94,0.08)",  label: "PENALTY" };
}

// ─── Single Toast ─────────────────────────────────────────────────────────────

const TOAST_LIFETIME = 3200; // ms

function RewardToastItem({ toast, onDismiss }: { toast: RewardToast; onDismiss: () => void }) {
  const [phase, setPhase] = useState<"enter" | "idle" | "exit">("enter");
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const palette = toastPalette(toast.reward);
  const meta = ACTION_META[toast.action];

  useEffect(() => {
    // Enter → idle
    const enterTimer = setTimeout(() => setPhase("idle"), 20);
    // idle → exit
    timerRef.current = setTimeout(() => setPhase("exit"), TOAST_LIFETIME - 400);
    // Exit → unmount
    const removeTimer = setTimeout(onDismiss, TOAST_LIFETIME);

    return () => {
      clearTimeout(enterTimer);
      if (timerRef.current) clearTimeout(timerRef.current);
      clearTimeout(removeTimer);
    };
  }, [onDismiss]);

  const rewardStr = `${toast.reward > 0 ? "+" : ""}${toast.reward.toFixed(1)}`;

  return (
    <div
      className={`rn-toast rn-toast--${phase}`}
      style={{
        "--border": palette.border,
        "--glow": palette.glow,
        "--accent": palette.accent,
        "--bg": palette.bg,
      } as React.CSSProperties}
      onClick={() => setPhase("exit")}
      role="status"
      aria-live="polite"
    >
      {/* Progress drain bar */}
      <div className="rn-toast__drain" style={{ animationDuration: `${TOAST_LIFETIME}ms` }} />

      <div className="rn-toast__body">
        <div className="rn-toast__icon">{meta.icon}</div>

        <div className="rn-toast__content">
          <div className="rn-toast__top">
            <span className="rn-toast__label" style={{ color: palette.accent }}>
              {palette.label}
            </span>
            <span className="rn-toast__step">Step {toast.step}</span>
          </div>
          <div className="rn-toast__reward" style={{ color: palette.accent }}>
            {rewardStr}
          </div>
          <div className="rn-toast__action">
            <span className="rn-toast__action-short">{meta.short}</span>
            <span className="rn-toast__action-name">{toast.action.replace(/_/g, " ")}</span>
          </div>
        </div>
      </div>

      <style>{`
        .rn-toast {
          position: relative;
          width: 220px;
          background: rgba(10,14,26,0.95);
          border: 1px solid var(--border);
          border-radius: 12px;
          overflow: hidden;
          cursor: pointer;
          box-shadow: 0 0 16px var(--glow), 0 4px 24px rgba(0,0,0,0.6);
          transition: transform 300ms cubic-bezier(0.34,1.56,0.64,1), opacity 300ms ease;
          backdrop-filter: blur(12px);
          background: var(--bg);
        }

        /* Entry / exit animation via phase classes */
        .rn-toast--enter {
          transform: translateX(calc(100% + 24px));
          opacity: 0;
        }
        .rn-toast--idle {
          transform: translateX(0);
          opacity: 1;
        }
        .rn-toast--exit {
          transform: translateX(calc(100% + 24px));
          opacity: 0;
        }

        .rn-toast__drain {
          position: absolute;
          bottom: 0;
          left: 0;
          height: 2px;
          width: 100%;
          background: var(--border);
          transform-origin: left;
          animation: drainBar linear forwards;
        }
        @keyframes drainBar {
          from { transform: scaleX(1); opacity: 1; }
          to   { transform: scaleX(0); opacity: 0.4; }
        }

        .rn-toast__body {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px 14px 14px;
        }

        .rn-toast__icon {
          font-size: 22px;
          color: var(--accent);
          line-height: 1;
          filter: drop-shadow(0 0 6px var(--glow));
          flex-shrink: 0;
        }

        .rn-toast__content { display: flex; flex-direction: column; gap: 3px; flex: 1; min-width: 0; }

        .rn-toast__top {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .rn-toast__label {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          font-weight: 700;
          letter-spacing: 0.14em;
        }
        .rn-toast__step {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          color: #4a5580;
        }

        .rn-toast__reward {
          font-family: 'JetBrains Mono', monospace;
          font-size: 26px;
          font-weight: 700;
          line-height: 1.1;
          letter-spacing: -0.02em;
          animation: rewardPop 400ms cubic-bezier(0.34,1.56,0.64,1) forwards;
        }
        @keyframes rewardPop {
          0%   { transform: scale(0.7); opacity: 0; }
          60%  { transform: scale(1.12); }
          100% { transform: scale(1);   opacity: 1; }
        }

        .rn-toast__action {
          display: flex;
          align-items: center;
          gap: 5px;
        }
        .rn-toast__action-short {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          padding: 1px 5px;
          border-radius: 3px;
          background: rgba(255,255,255,0.06);
          color: var(--accent);
          letter-spacing: 0.08em;
        }
        .rn-toast__action-name {
          font-family: 'DM Sans', system-ui, sans-serif;
          font-size: 11px;
          color: #8899cc;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
      `}</style>
    </div>
  );
}

// ─── Stack (Portal) ───────────────────────────────────────────────────────────

function RewardNotificationStack() {
  const { toasts, dismiss } = useToastContext();

  // Auto-register notifier when stack mounts
  useRewardNotifier();

  return (
    <div
      className="rn-stack"
      role="region"
      aria-label="Reward notifications"
    >
      {toasts.map((toast) => (
        <RewardToastItem
          key={toast.id}
          toast={toast}
          onDismiss={() => dismiss(toast.id)}
        />
      ))}

      <style>{`
        .rn-stack {
          position: fixed;
          bottom: 24px;
          right: 24px;
          z-index: 9999;
          display: flex;
          flex-direction: column-reverse;
          gap: 10px;
          pointer-events: none;
        }
        .rn-stack > * { pointer-events: all; }
      `}</style>
    </div>
  );
}

// ─── Manual trigger hook ──────────────────────────────────────────────────────
// Expose for components that want to fire a toast directly (e.g., ActionPanel).

export function useFireReward() {
  const { push } = useToastContext();
  return useCallback(
    (reward: number, action: Action, step: number) => {
      push({ reward, action, step, timestamp: Date.now() });
    },
    [push]
  );
}

export default RewardNotificationStack;
