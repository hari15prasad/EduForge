// lib/dqn-client.ts
// Mock client that simulates dqn_pipeline.py responses.

export type HeadType     = "FACTUAL" | "PROCEDURAL" | "CONCEPTUAL" | "TRANSFER";
export type ActionType   = "explain" | "correct_fact" | "worked_example" | "analogize" | "question";

export interface StudentState {
  confusion:           number; // 0–10
  attention:           number; // 0–10
  domain_index:        number; // 0–3
  step_count:          number; // 0–20
  last_action_impact:  number; // -1 to 1
}

export interface ForwardResult {
  action:          ActionType;
  selected_head:   HeadType;
  q_values:        Record<ActionType, number>;
  expected_reward: number;
  next_state:      StudentState;
  confusion_delta: number;
  attention_delta: number;
  terminated:      boolean;
  termination_reason?: "success" | "timeout";
  generated_text?: string;
  handling_score?: number;
}

export interface EpisodeSummary {
  episode:       number;
  head:          HeadType;
  total_reward:  number;
  steps:         number;
  success:       boolean;
}

export interface TrainingStats {
  head:          HeadType;
  episodeRewards: number[];
  successRate:   number;
  avgSteps:      number;
  totalEpisodes: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const ACTIONS: ActionType[] = [
  "explain", "correct_fact", "worked_example", "analogize", "question",
];

const HEADS: HeadType[] = ["FACTUAL", "PROCEDURAL", "CONCEPTUAL", "TRANSFER"];

const DOMAIN_HEAD: Record<number, HeadType> = {
  0: "FACTUAL", 1: "PROCEDURAL", 2: "CONCEPTUAL", 3: "TRANSFER",
};

/** Action → expected confusion reduction per domain (rough sim) */
const ACTION_EFFICACY: Record<ActionType, Record<HeadType, number>> = {
  explain:        { FACTUAL: 1.8, PROCEDURAL: 0.8, CONCEPTUAL: 1.2, TRANSFER: 0.6 },
  correct_fact:   { FACTUAL: 2.2, PROCEDURAL: 0.6, CONCEPTUAL: 0.4, TRANSFER: 0.3 },
  worked_example: { FACTUAL: 0.4, PROCEDURAL: 2.4, CONCEPTUAL: 1.0, TRANSFER: 0.8 },
  analogize:      { FACTUAL: 0.6, PROCEDURAL: 0.8, CONCEPTUAL: 2.0, TRANSFER: 1.8 },
  question:       { FACTUAL: 0.8, PROCEDURAL: 1.0, CONCEPTUAL: 1.4, TRANSFER: 2.2 },
};

function rand(min: number, max: number) {
  return Math.random() * (max - min) + min;
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function jitter(base: number, noise = 0.4): number {
  return base + rand(-noise, noise);
}

/** Simulate the multi-head Q-network choosing an action. */
function selectAction(state: StudentState): { action: ActionType; qValues: Record<ActionType, number> } {
  const head = DOMAIN_HEAD[state.domain_index];

  // Generate Q-values: efficacy-based with noise + confusion/attention modulation
  const qValues = {} as Record<ActionType, number>;
  for (const a of ACTIONS) {
    const base = ACTION_EFFICACY[a][head];
    const confusionBonus  = (state.confusion / 10) * 0.5;
    const attentionPenalty = state.attention < 4 ? -0.5 : 0;
    qValues[a] = jitter(base + confusionBonus + attentionPenalty, 0.5);
  }

  // ε-greedy: 10 % random exploration
  const action: ActionType =
    Math.random() < 0.10
      ? ACTIONS[Math.floor(Math.random() * ACTIONS.length)]
      : (Object.entries(qValues).sort(([, a], [, b]) => b - a)[0][0] as ActionType);

  return { action, qValues };
}

/** Simulate environment step. */
function stepEnv(state: StudentState, action: ActionType): {
  nextState: StudentState;
  reward: number;
  confusionDelta: number;
  attentionDelta: number;
  terminated: boolean;
  terminationReason?: "success" | "timeout";
} {
  const head = DOMAIN_HEAD[state.domain_index];
  const efficacy = ACTION_EFFICACY[action][head];

  const confusionDelta = -jitter(efficacy * 0.7, 0.3);
  const attentionDelta  = jitter(efficacy * 0.2 - 0.1, 0.2);

  const newConfusion  = clamp(state.confusion + confusionDelta, 0, 10);
  const newAttention  = clamp(state.attention + attentionDelta, 0, 10);
  const newStep       = state.step_count + 1;
  const impact        = clamp(-(confusionDelta / 2), -1, 1);

  let reward = -1; // step penalty
  if (newAttention < 4.0) reward += -10;

  let terminated = false;
  let terminationReason: "success" | "timeout" | undefined;

  if (newConfusion <= 2.0) {
    reward += 100;
    terminated = true;
    terminationReason = "success";
  } else if (newStep >= 20) {
    terminated = true;
    terminationReason = "timeout";
  }

  const nextState: StudentState = {
    confusion:          newConfusion,
    attention:          newAttention,
    domain_index:       state.domain_index,
    step_count:         newStep,
    last_action_impact: impact,
  };

  return { nextState, reward, confusionDelta, attentionDelta, terminated, terminationReason };
}

// ── Canned tutor messages ─────────────────────────────────────────────────────

const TUTOR_MESSAGES: Record<ActionType, string[]> = {
  explain: [
    "Let me walk you through this concept step by step. First, we identify the core principle, then apply it directly to your situation.",
    "Here's a clear breakdown of the underlying principle: ensure that all your variables and states are properly aligned before proceeding.",
    "The key insight is how these components interact. Always look at the relationship between the inputs and the expected outputs.",
  ],
  correct_fact: [
    "There's a small factual slip there. The correct version involves fully evaluating the expression rather than stopping halfway.",
    "That's a common misconception. The accurate way to approach this is to rely on the established base rules.",
    "Good try! To refine the details: make sure you account for all edge cases in your assumption.",
  ],
  worked_example: [
    "Let's trace through a concrete example together. For instance, if your input is X, applying the function yields Y.",
    "Here's a worked solution so you can see the pattern: Step 1 isolates the variable, Step 2 solves it.",
    "Follow each step in this example — notice where the logic shifts to handle the new constraint.",
  ],
  analogize: [
    "Think of it like a sorting algorithm choosing its pivot. The efficiency depends heavily on that initial choice.",
    "It's analogous to gradient descent finding a valley. You iteratively take steps to minimize the error.",
    "Imagine the state space as a complex chess board. Each move constrains your future possibilities.",
  ],
  question: [
    "Before we proceed — how would you apply this to a slightly different scenario?",
    "Can you predict what the outcome would be if we changed the initial parameters?",
    "What would happen if we removed that specific constraint?",
  ],
};

function pickMessage(action: ActionType): string {
  const pool = TUTOR_MESSAGES[action];
  return pool[Math.floor(Math.random() * pool.length)];
}

// ── Hybrid backend call ──────────────────────────────────────────────────────

async function callHybridBackend(
  state: StudentState,
  userText: string,
  forcedStrategy?: string,
  sessionId: string = "default",
  signal?: AbortSignal
): Promise<{ strategy: string; q_values: Record<number, number>; response: string; handling_score: number } | null> {
  try {
    const domainIdx = state.domain_index;
    const stateVector = [
      state.confusion / 10,
      state.attention / 10,
      state.step_count / 20,
      domainIdx === 0 ? 1 : 0,
      domainIdx === 1 ? 1 : 0,
      domainIdx === 2 ? 1 : 0,
      domainIdx === 3 ? 1 : 0,
      state.last_action_impact,
      0, 0, 0, 0, 0,  // history padding
    ];

    const res = await fetch("/api/hybrid/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        student_message: userText,
        state_vector: stateVector,
        domain_idx: domainIdx,
        forced_strategy: forcedStrategy,
        session_id: sessionId,
      }),
    });

    if (!res.ok) return null;
    return await res.json();
  } catch (e: any) {
    if (e.name === 'AbortError') throw e;
    return null;
  }
}

// ── Smart Tutoring Response Engine ───────────────────────────────────────────
// Used as a fallback when the backend is unavailable.
// Generates real, helpful tutoring answers instead of generic canned text.

function generateTutoringResponse(userText: string, action: ActionType): string {
  const q = userText.trim();
  const lower = q.toLowerCase();

  // ── Greetings ────────────────────────────────────────────────────────────
  if (/^(hi|hello|hey|sup|greetings|howdy)\b/i.test(lower)) {
    return "Hello! I'm EduForge, your AI tutor. What topic would you like to explore today? I can help with math, coding, science, and more!";
  }

  // ── Arithmetic ───────────────────────────────────────────────────────────
  const sqrtMatch = lower.match(/square\s*root\s*of\s*(\d+(\.\d+)?)/);
  if (sqrtMatch) {
    const n = parseFloat(sqrtMatch[1]);
    return `The square root of ${n} is **${Math.sqrt(n).toFixed(4)}**.\n\nTo verify: ${Math.sqrt(n).toFixed(4)} × ${Math.sqrt(n).toFixed(4)} ≈ ${n}.`;
  }

  const addMatch = lower.match(/(\d+)\s*\+\s*(\d+)/);
  if (addMatch) {
    const result = parseInt(addMatch[1]) + parseInt(addMatch[2]);
    return `${addMatch[1]} + ${addMatch[2]} = **${result}**.`;
  }

  const mulMatch = lower.match(/(\d+)\s*[×x\*]\s*(\d+)/);
  if (mulMatch) {
    const result = parseInt(mulMatch[1]) * parseInt(mulMatch[2]);
    return `${mulMatch[1]} × ${mulMatch[2]} = **${result}**.`;
  }

  // ── Binomial / Expansion ─────────────────────────────────────────────────
  if (lower.includes("binomial")) {
    const nMatch = lower.match(/n\s*=\s*(\d+)/);
    const nVal = nMatch ? parseInt(nMatch[1]) : null;
    let resp = "The **Binomial Theorem** states that:\n\n**(a + b)ⁿ = Σ C(n,k) · aⁿ⁻ᵏ · bᵏ** for k = 0 to n\n\nwhere C(n,k) = n! / (k!(n-k)!) is the binomial coefficient.";
    if (nVal !== null) {
      resp += `\n\nFor **n = ${nVal}**, the expansion of (a+b)${nVal} has ${nVal + 1} terms:`;
      for (let k = 0; k <= nVal; k++) {
        const coef = factorial(nVal) / (factorial(k) * factorial(nVal - k));
        resp += `\n  • k=${k}: **${coef}** · a^${nVal - k} · b^${k}`;
      }
    }
    return resp;
  }

  // ── Variables ────────────────────────────────────────────────────────────
  if (/\\b(variable|variables)\\b/.test(lower)) {
    return "A **variable** is a named container that stores a value in memory.\\n\\n```python\\nstudent_name = 'Alice'   # stores text\\nscore = 95               # stores a number\\nis_passing = True        # stores true/false\\n```\\n\\nYou can change the value any time — that's why it's called a *variable* (it can vary).";
  }

  // ── Loops ────────────────────────────────────────────────────────────────
  if (/\\b(loop|loops|for loop|while loop)\\b/.test(lower)) {
    return "A **loop** repeats a block of code automatically.\\n\\n**For loop** — when you know how many times:\\n```python\\nfor i in range(5):\\n    print(i)  # prints 0, 1, 2, 3, 4\\n```\\n\\n**While loop** — when you repeat until a condition is false:\\n```python\\ncount = 0\\nwhile count < 5:\\n    print(count)\\n    count += 1\\n```\\n\\nWhich type of loop are you working with?";
  }

  // ── Functions / Methods ──────────────────────────────────────────────────
  if (/\\b(function|functions|def)\\b/.test(lower)) {
    return "A **function** is a reusable block of code that does one specific job.\\n\\n```python\\ndef greet(name):\\n    return f'Hello, {name}!'\\n\\nprint(greet('Alice'))  # Hello, Alice!\\n```\\n\\nFunctions help you avoid repeating code — write once, call many times. What do you want your function to do?";
  }

  // ── Recursion ────────────────────────────────────────────────────────────
  if (/\\b(recursion|recursive)\\b/.test(lower)) {
    return "**Recursion** is when a function calls itself to solve a smaller version of the same problem.\\n\\n```python\\ndef factorial(n):\\n    if n == 0:       # base case — stops the recursion\\n        return 1\\n    return n * factorial(n - 1)  # recursive call\\n\\nprint(factorial(5))  # 120\\n```\\n\\nEvery recursive solution needs: (1) a **base case** to stop, (2) a **recursive step** that moves toward it.";
  }

  // ── Sorting ──────────────────────────────────────────────────────────────
  if (/\\b(sort|sorting|bubble sort|merge sort|quick sort)\\b/.test(lower)) {
    return "**Sorting** algorithms arrange data in order. The most common:\\n\\n| Algorithm | Best | Worst | Space |\\n|---|---|---|---|\\n| Bubble Sort | O(n) | O(n²) | O(1) |\\n| Merge Sort | O(n log n) | O(n log n) | O(n) |\\n| Quick Sort | O(n log n) | O(n²) | O(log n) |\\n\\nIn Python, use `sorted(list)` for a new sorted list, or `list.sort()` to sort in place. Which algorithm would you like to understand deeper?";
  }

  // ── OOP / Classes ────────────────────────────────────────────────────────
  if (/\\b(oop|object oriented)\\b/.test(lower) || ( /\\b(class|object)\\b/.test(lower) && /\\b(programming|code|python|java|c\\+\\+|oop)\\b/.test(lower) )) {
    return "**Object-Oriented Programming (OOP)** models real-world entities as objects.\\n\\n```python\\nclass Student:\\n    def __init__(self, name, grade):\\n        self.name = name\\n        self.grade = grade\\n\\n    def is_passing(self):\\n        return self.grade >= 60\\n\\ns = Student('Alice', 85)\\nprint(s.is_passing())  # True\\n```\\n\\nKey ideas: **Class** = blueprint, **Object** = instance, **Method** = function inside a class. What aspect of OOP are you exploring?";
  }

  // ── Strategy-based generic response ──────────────────────────────────────
  const strategyResponses: Record<ActionType, string> = {
    explain:        `Great question! Let me break down **"${q}"** clearly.\n\n${pickMessage("explain")}\n\nDo you want me to go deeper into any part of this?`,
    correct_fact:   `Let's make sure we have the right foundation for **"${q}"**.\n\n${pickMessage("correct_fact")}\n\nWhat specifically are you unsure about?`,
    worked_example: `The best way to understand **"${q}"** is through an example.\n\n${pickMessage("worked_example")}\n\nWould you like to try a similar problem yourself?`,
    analogize:      `Think of **"${q}"** this way:\n\n${pickMessage("analogize")}\n\nDoes that analogy help clarify things?`,
    question:       `Before I answer **"${q}"**, let me ask you: ${pickMessage("question")}\n\nYour thinking will help me tailor the best explanation.`,
  };

  return strategyResponses[action];
}

function factorial(n: number): number {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

// ── Public client ─────────────────────────────────────────────────────────────


export class DQNClient {
  private _episodeLog: EpisodeSummary[] = [];
  private _episodeCount = 0;
  private _currentAbortController: AbortController | null = null;

  abortCurrent() {
    if (this._currentAbortController) {
      this._currentAbortController.abort();
      this._currentAbortController = null;
    }
  }

  /**
   * Single forward pass + environment step.
   * Call this on each chat turn.
   */
  async forward(state: StudentState, userText?: string, forcedStrategy?: ActionType, sessionId: string = "default"): Promise<ForwardResult> {
    // When user sends a message, try the real backend first
    if (userText || forcedStrategy) {
      // Map ActionType to backend strategy name
      const ACTION_TO_STRATEGY: Record<ActionType, string> = {
        explain:        "EXPLAIN",
        worked_example: "WORKED_EXAMPLE",
        analogize:      "ANALOGIZE",
        question:       "QUESTION",
        correct_fact:   "CORRECT_FACT",
      };

      let backendResult;
      this._currentAbortController = new AbortController();
      try {
        backendResult = await callHybridBackend(
          state, 
          userText || `The tutor chosen the ${forcedStrategy} strategy.`,
          forcedStrategy ? ACTION_TO_STRATEGY[forcedStrategy] : undefined,
          sessionId,
          this._currentAbortController?.signal
        );
      } catch (e: any) {
        if (e.name === 'AbortError') throw e;
      } finally {
        this._currentAbortController = null;
      }

      if (backendResult) {
        // Map backend strategy name to ActionType
        const STRATEGY_TO_ACTION: Record<string, ActionType> = {
          EXPLAIN:           "explain",
          WORKED_EXAMPLE:    "worked_example",
          ANALOGIZE:         "analogize",
          SOCRATIC_QUESTION: "question",
          CORRECT_FACT:      "correct_fact",
        };
        const action = STRATEGY_TO_ACTION[backendResult.strategy] ?? "explain";
        const { nextState, reward, confusionDelta, attentionDelta, terminated, terminationReason } = stepEnv(state, action);
        const selected_head = DOMAIN_HEAD[state.domain_index];
        if (terminated) {
          this._episodeCount += 1;
          this._episodeLog.push({ episode: this._episodeCount, head: selected_head, total_reward: reward, steps: nextState.step_count, success: terminationReason === "success" });
        }
        return {
          action, selected_head,
          q_values: Object.fromEntries(ACTIONS.map((a, i) => [a, backendResult.q_values[i] ?? 0])) as Record<ActionType, number>,
          expected_reward: reward,
          next_state: nextState,
          confusion_delta: confusionDelta,
          attention_delta: attentionDelta,
          terminated,
          termination_reason: terminationReason,
          generated_text:  backendResult.response,
          handling_score:  backendResult.handling_score,
        };
      }
    }

    // Fallback: local simulation (auto-pilot or backend unavailable)
    const latency = userText ? 400 + Math.random() * 300 : 60 + Math.random() * 80;
    await new Promise((r) => setTimeout(r, latency));

    const { action, qValues }  = selectAction(state);
    const {
      nextState, reward, confusionDelta, attentionDelta,
      terminated, terminationReason,
    } = stepEnv(state, action);

    const selected_head = DOMAIN_HEAD[state.domain_index];

    if (terminated) {
      this._episodeCount += 1;
      this._episodeLog.push({
        episode:      this._episodeCount,
        head:         selected_head,
        total_reward: reward,
        steps:        nextState.step_count,
        success:      terminationReason === "success",
      });
    }
    
    // Generate a real tutoring response when backend is unavailable
    let generated_text;
    let handling_score;
    if (userText) {
      handling_score = 65;
      generated_text = generateTutoringResponse(userText, action);
    }


    return {
      action,
      selected_head,
      q_values:        qValues,
      expected_reward: reward,
      next_state:      nextState,
      confusion_delta: confusionDelta,
      attention_delta: attentionDelta,
      terminated,
      termination_reason: terminationReason,
      generated_text,
      handling_score
    };
  }

  /**
   * Returns per-head training statistics derived from the episode log,
   * seeded with synthetic history for a richer initial UI.
   */
  getTrainingStats(): TrainingStats[] {
    return HEADS.map((head) => {
      const episodes = this._episodeLog.filter((e) => e.head === head);

      // Synthetic baseline: 40 warm-start episodes
      const syntheticRewards = Array.from({ length: 40 }, (_, i) =>
        -20 + i * 3.2 + rand(-8, 8)
      );

      const allRewards = [...syntheticRewards, ...episodes.map((e) => e.total_reward)];
      const successes  = episodes.filter((e) => e.success).length;
      const totalEps   = 40 + episodes.length;

      return {
        head,
        episodeRewards: allRewards,
        successRate:    episodes.length > 0 ? successes / episodes.length : rand(0.5, 0.85),
        avgSteps:       episodes.length > 0
          ? episodes.reduce((s, e) => s + e.steps, 0) / episodes.length
          : rand(8, 16),
        totalEpisodes:  totalEps,
      };
    });
  }

  /** Returns a canned tutor message for the chosen action. */
  getTutorMessage(action: ActionType): string {
    return pickMessage(action);
  }

  /** Reset episode log (new training run). */
  reset() {
    this._episodeLog  = [];
    this._episodeCount = 0;
  }
}

// Singleton export
export const dqnClient = new DQNClient();
