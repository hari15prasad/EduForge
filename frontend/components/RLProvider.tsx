"use client";

import React, {
  createContext,
  useContext,
  useReducer,
  useCallback,
  useMemo,
  ReactNode,
} from "react";

// ─── Types ───────────────────────────────────────────────────────────────────

export type Domain = "Factual" | "Procedural" | "Conceptual" | "Transfer";
export type AgentStatus = "online" | "training" | "evaluating" | "idle";
export type Action = "explain" | "correct_fact" | "worked_example" | "analogize" | "question";

export interface RewardEntry {
  step: number;
  reward: number;
  action: Action;
  timestamp: number;
}

export interface EpisodeStats {
  episodeId: number;
  totalReward: number;
  steps: number;
  outcome: "success" | "timeout" | "in_progress";
  domain: Domain;
}

export interface RLState {
  // Core student state
  confusion: number;       // 0–10
  attention: number;       // 0–10
  domainIndex: number;     // 0–3
  currentStep: number;     // 0–20
  lastActionImpact: number;

  // Agent metadata
  agentStatus: AgentStatus;
  episodeCount: number;
  epsilon: number;         // exploration rate 0–1

  // History
  rewardHistory: RewardEntry[];
  episodeHistory: EpisodeStats[];
  actionLog: { action: Action; step: number }[];

  // Derived / computed
  currentDomain: Domain;
  isTerminated: boolean;
  cumulativeReward: number;
}

// ─── Initial State ────────────────────────────────────────────────────────────

const DOMAINS: Domain[] = ["Factual", "Procedural", "Conceptual", "Transfer"];

const initialState: RLState = {
  confusion: 7.0,
  attention: 6.0,
  domainIndex: 0,
  currentStep: 0,
  lastActionImpact: 0,

  agentStatus: "idle",
  episodeCount: 0,
  epsilon: 1.0,

  rewardHistory: [],
  episodeHistory: [],
  actionLog: [],

  currentDomain: "Factual",
  isTerminated: false,
  cumulativeReward: 0,
};

// ─── Actions ──────────────────────────────────────────────────────────────────

type RLAction =
  | { type: "SET_CONFUSION"; payload: number }
  | { type: "SET_ATTENTION"; payload: number }
  | { type: "SET_DOMAIN"; payload: number }
  | { type: "SET_STEP"; payload: number }
  | { type: "SET_AGENT_STATUS"; payload: AgentStatus }
  | { type: "SET_EPSILON"; payload: number }
  | { type: "PUSH_REWARD"; payload: RewardEntry }
  | { type: "PUSH_EPISODE"; payload: EpisodeStats }
  | { type: "LOG_ACTION"; payload: { action: Action; step: number } }
  | { type: "STEP_UPDATE"; payload: Partial<Pick<RLState, "confusion" | "attention" | "domainIndex" | "currentStep" | "lastActionImpact">> & { reward: RewardEntry } }
  | { type: "RESET_EPISODE" }
  | { type: "RESET_ALL" };

// ─── Reducer ──────────────────────────────────────────────────────────────────

function rlReducer(state: RLState, action: RLAction): RLState {
  switch (action.type) {
    case "SET_CONFUSION":
      return {
        ...state,
        confusion: Math.max(0, Math.min(10, action.payload)),
        isTerminated: action.payload <= 2.0 || state.currentStep >= 20,
      };

    case "SET_ATTENTION":
      return { ...state, attention: Math.max(0, Math.min(10, action.payload)) };

    case "SET_DOMAIN":
      return {
        ...state,
        domainIndex: action.payload,
        currentDomain: DOMAINS[action.payload] ?? "Factual",
      };

    case "SET_STEP":
      return {
        ...state,
        currentStep: action.payload,
        isTerminated: state.confusion <= 2.0 || action.payload >= 20,
      };

    case "SET_AGENT_STATUS":
      return { ...state, agentStatus: action.payload };

    case "SET_EPSILON":
      return { ...state, epsilon: Math.max(0, Math.min(1, action.payload)) };

    case "PUSH_REWARD": {
      const updated = [...state.rewardHistory, action.payload];
      return {
        ...state,
        rewardHistory: updated,
        cumulativeReward: state.cumulativeReward + action.payload.reward,
      };
    }

    case "PUSH_EPISODE":
      return {
        ...state,
        episodeHistory: [...state.episodeHistory, action.payload],
        episodeCount: state.episodeCount + 1,
      };

    case "LOG_ACTION":
      return {
        ...state,
        actionLog: [...state.actionLog, action.payload],
      };

    case "STEP_UPDATE": {
      const { reward, ...studentUpdate } = action.payload;
      const newConfusion = studentUpdate.confusion ?? state.confusion;
      const newStep = studentUpdate.currentStep ?? state.currentStep;
      const newDomainIndex = studentUpdate.domainIndex ?? state.domainIndex;
      const updatedRewards = [...state.rewardHistory, reward];

      return {
        ...state,
        ...studentUpdate,
        currentDomain: DOMAINS[newDomainIndex] ?? state.currentDomain,
        rewardHistory: updatedRewards,
        cumulativeReward: state.cumulativeReward + reward.reward,
        isTerminated: newConfusion <= 2.0 || newStep >= 20,
      };
    }

    case "RESET_EPISODE":
      return {
        ...state,
        confusion: 7.0 + Math.random() * 2,
        attention: 5.0 + Math.random() * 3,
        currentStep: 0,
        lastActionImpact: 0,
        cumulativeReward: 0,
        isTerminated: false,
        rewardHistory: [],
        actionLog: [],
      };

    case "RESET_ALL":
      return { ...initialState };

    default:
      return state;
  }
}

// ─── Context ──────────────────────────────────────────────────────────────────

interface RLContextValue {
  state: RLState;

  // Setters
  setConfusion: (v: number) => void;
  setAttention: (v: number) => void;
  setDomain: (index: number) => void;
  setStep: (step: number) => void;
  setAgentStatus: (status: AgentStatus) => void;
  setEpsilon: (e: number) => void;

  // Complex updates
  pushReward: (entry: RewardEntry) => void;
  pushEpisode: (stats: EpisodeStats) => void;
  logAction: (action: Action, step: number) => void;
  applyStepUpdate: (
    studentUpdate: Partial<Pick<RLState, "confusion" | "attention" | "domainIndex" | "currentStep" | "lastActionImpact">>,
    reward: RewardEntry
  ) => void;

  // Episode control
  resetEpisode: () => void;
  resetAll: () => void;

  // Computed helpers
  domains: readonly Domain[];
  recentRewards: RewardEntry[];
  avgRecentReward: number;
}

const RLContext = createContext<RLContextValue | null>(null);

// ─── Provider ─────────────────────────────────────────────────────────────────

export function RLProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(rlReducer, initialState);

  const setConfusion = useCallback((v: number) => dispatch({ type: "SET_CONFUSION", payload: v }), []);
  const setAttention = useCallback((v: number) => dispatch({ type: "SET_ATTENTION", payload: v }), []);
  const setDomain = useCallback((index: number) => dispatch({ type: "SET_DOMAIN", payload: index }), []);
  const setStep = useCallback((step: number) => dispatch({ type: "SET_STEP", payload: step }), []);
  const setAgentStatus = useCallback((status: AgentStatus) => dispatch({ type: "SET_AGENT_STATUS", payload: status }), []);
  const setEpsilon = useCallback((e: number) => dispatch({ type: "SET_EPSILON", payload: e }), []);

  const pushReward = useCallback((entry: RewardEntry) => dispatch({ type: "PUSH_REWARD", payload: entry }), []);
  const pushEpisode = useCallback((stats: EpisodeStats) => dispatch({ type: "PUSH_EPISODE", payload: stats }), []);
  const logAction = useCallback((action: Action, step: number) => dispatch({ type: "LOG_ACTION", payload: { action, step } }), []);

  const applyStepUpdate = useCallback(
    (
      studentUpdate: Partial<Pick<RLState, "confusion" | "attention" | "domainIndex" | "currentStep" | "lastActionImpact">>,
      reward: RewardEntry
    ) => dispatch({ type: "STEP_UPDATE", payload: { ...studentUpdate, reward } }),
    []
  );

  const resetEpisode = useCallback(() => dispatch({ type: "RESET_EPISODE" }), []);
  const resetAll = useCallback(() => dispatch({ type: "RESET_ALL" }), []);

  const recentRewards = useMemo(() => state.rewardHistory.slice(-20), [state.rewardHistory]);
  const avgRecentReward = useMemo(
    () => recentRewards.length ? recentRewards.reduce((s, r) => s + r.reward, 0) / recentRewards.length : 0,
    [recentRewards]
  );

  const value: RLContextValue = useMemo(() => ({
    state,
    setConfusion, setAttention, setDomain, setStep, setAgentStatus, setEpsilon,
    pushReward, pushEpisode, logAction, applyStepUpdate,
    resetEpisode, resetAll,
    domains: DOMAINS,
    recentRewards,
    avgRecentReward,
  }), [
    state,
    setConfusion, setAttention, setDomain, setStep, setAgentStatus, setEpsilon,
    pushReward, pushEpisode, logAction, applyStepUpdate,
    resetEpisode, resetAll,
    recentRewards, avgRecentReward,
  ]);

  return <RLContext.Provider value={value}>{children}</RLContext.Provider>;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useRL(): RLContextValue {
  const ctx = useContext(RLContext);
  if (!ctx) throw new Error("useRL must be used within <RLProvider>");
  return ctx;
}

// ─── Selector Hooks ───────────────────────────────────────────────────────────

export const useConfusion = () => useRL().state.confusion;
export const useAttention = () => useRL().state.attention;
export const useDomainIndex = () => useRL().state.domainIndex;
export const useCurrentStep = () => useRL().state.currentStep;
export const useRewardHistory = () => useRL().state.rewardHistory;
export const useAgentStatus = () => useRL().state.agentStatus;
export const useEpisodeHistory = () => useRL().state.episodeHistory;
export const useCurrentDomain = () => useRL().state.currentDomain;
export const useIsTerminated = () => useRL().state.isTerminated;
