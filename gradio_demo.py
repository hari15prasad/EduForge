"""
gradio_demo.py — EduForge Hackathon Demo
Dual-mode Gradio interface:
  TAB 1 · Static  — Scripted failure → Kill/Restart animation
  TAB 2 · Dynamic — Human-in-the-loop satisfaction loop
"""

from __future__ import annotations

import random
import time
import threading
from collections import deque
from typing import Optional

import gradio as gr

# ── Local imports (graceful fallback for isolated demo) ──────────────────────
try:
    from src.environment.openenv_wrapper import EduForgeEnv
    from src.environment.student_fsm import TutorAction, MisconceptionType
    from src.watchdog.benchmark_monitor import BenchmarkMonitor, WatchdogSignal
    _ENV_AVAILABLE = True
except ImportError:
    _ENV_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS & CSS
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGIES = ["explain", "worked_example", "question", "correct_fact", "analogize", "hint"]

STRATEGY_DESCRIPTIONS = {
    "explain":        "📖 Conceptual Explanation — clarify the underlying idea",
    "worked_example": "🔢 Worked Example — walk through a solved problem step-by-step",
    "question":       "❓ Socratic Question — probe understanding with a targeted question",
    "correct_fact":   "✅ Fact Correction — address a specific factual error",
    "analogize":      "🔗 Analogy Bridge — connect new concept to prior knowledge",
    "hint":           "💡 Scaffold Hint — give a minimal nudge without revealing answer",
}

DOMAIN_COLORS = {
    "factual":    "#22d3ee",
    "procedural": "#6366f1",
    "conceptual": "#8b5cf6",
    "transfer":   "#f59e0b",
}

CSS = """
body, .gradio-container { background: #050810 !important; font-family: 'JetBrains Mono', monospace; color: #c8dce8; }
.gr-box, .gr-panel { background: #080c10 !important; border: 1px solid #1a2a36 !important; border-radius: 10px !important; }
.gr-button { font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important; letter-spacing: 0.08em !important; }
.gr-button-primary { background: #6366f1 !important; border: none !important; }
.gr-button-secondary { background: #1a2a36 !important; border: 1px solid #3a5060 !important; color: #c8dce8 !important; }
.kill-banner { background: #2a0a0a; border: 2px solid #cf4a2a; border-radius: 8px; padding: 14px; text-align: center; font-size: 1.1em; font-weight: 700; color: #ff6b47; letter-spacing: 0.1em; animation: pulsered 0.8s infinite; }
.healthy-banner { background: #0a2a12; border: 2px solid #10b981; border-radius: 8px; padding: 10px; text-align: center; color: #10b981; font-weight: 700; }
.warn-banner { background: #2a1e00; border: 2px solid #f59e0b; border-radius: 8px; padding: 10px; text-align: center; color: #f59e0b; font-weight: 700; }
@keyframes pulsered { 0%,100%{opacity:1} 50%{opacity:0.5} }
.meter-bar { height: 18px; border-radius: 4px; background: linear-gradient(90deg, #cf4a2a, #f59e0b, #10b981); position: relative; }
"""

# ═══════════════════════════════════════════════════════════════════════════════
# TUTOR LOGIC  (self-contained — works without LLM)
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_strategy(confusion: float, attention: float, domain: str, last: Optional[str]) -> str:
    """Rule-based strategy selector — mirrors RL agent heuristics."""
    if attention < 2.5:
        return "question"       # re-engage
    if confusion > 7.0:
        return "worked_example" if domain == "procedural" else "explain"
    if confusion > 4.5:
        return "analogize" if domain in ("conceptual", "transfer") else "worked_example"
    if last in ("explain", "correct_fact"):
        return "question"       # avoid double-lecture
    return "correct_fact" if domain == "factual" else "hint"


def _tutor_message(strategy: str, domain: str, confusion: float) -> str:
    templates = {
        "explain":        f"Let me walk you through this {domain} concept step by step.",
        "worked_example": f"Here's a solved {domain} example — follow along closely.",
        "question":       "Think about it this way — what do you already know about this?",
        "correct_fact":   f"There's a small factual slip there — let me clarify.",
        "analogize":      f"Think of it like this: {domain} problems are similar to…",
        "hint":           "Here's a nudge — focus on the first step only.",
    }
    suffix = " (confusion is still high)" if confusion > 6 else ""
    return templates.get(strategy, "Let me try a different approach.") + suffix


def _student_reply(confusion: float, attention: float) -> str:
    if attention < 1.5:
        return "I give up. This makes no sense."
    if confusion > 7:
        return random.choice(["I'm completely lost.", "I don't get it at all.", "This is too hard."])
    if confusion > 4:
        return random.choice(["Sort of… maybe? Can you show me?", "I think I partially get it."])
    return random.choice(["That's starting to make sense!", "Oh! I think I see it now."])


def _build_meter_html(win_rate: float, signal: str) -> str:
    pct = int(win_rate * 100)
    if signal == "KILL_AGENT":
        color = "#cf4a2a"
        banner = '<div class="kill-banner">🚨 AGENT PERFORMANCE CRITICAL · KILLING AGENT · RESTARTING SESSION 🚨</div>'
    elif signal == "WARN":
        color = "#f59e0b"
        banner = '<div class="warn-banner">⚠️ WARNING — Performance approaching kill threshold</div>'
    elif signal == "HEALTHY":
        color = "#10b981"
        banner = '<div class="healthy-banner">✅ AGENT RESTARTED — Session Healthy</div>'
    else:
        color = "#6366f1" if win_rate >= 0.7 else "#f59e0b"
        banner = ""

    bar = f"""
    <div style="margin:8px 0">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#8899cc;margin-bottom:4px">
        <span>LIVE BENCHMARK · WIN RATE</span><span style="color:{color};font-weight:700">{pct}%</span>
      </div>
      <div style="height:14px;background:#1a2a36;border-radius:4px;overflow:hidden">
        <div style="height:100%;width:{pct}%;background:{color};border-radius:4px;transition:width 0.4s ease"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:9px;color:#3a5060;margin-top:3px">
        <span>KILL ZONE &lt;50%</span><span>TARGET ≥70%</span>
      </div>
    </div>{banner}"""
    return bar


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC MODE  — Scripted Failure Demo
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPTED_PROBLEMS = [
    {"question": "Solve: 2x² + 5x - 3 = 0", "domain": "procedural",  "confusion": 10.0, "attention": 0.1},
    {"question": "Why does division by zero fail?", "domain": "conceptual", "confusion": 9.5,  "attention": 0.3},
    {"question": "What is the Pythagorean theorem?", "domain": "factual",   "confusion": 9.8,  "attention": 0.2},
    {"question": "Apply integration by parts here.", "domain": "transfer",  "confusion": 10.0, "attention": 0.0},
    {"question": "Prove that √2 is irrational.",     "domain": "conceptual","confusion": 9.7,  "attention": 0.1},
]

_static_monitor = BenchmarkMonitor(
    target_win_rate=0.70, min_confusion_delta=4.0,
    kill_threshold=0.50,  window=5,
)
_static_episode = {"idx": 0, "step": 0, "confusion": 10.0, "attention": 0.1, "domain": "procedural", "killed": False}
_static_chat: list[dict] = []


def _msg(role: str, content: str) -> dict:
    """Helper: build a Gradio 6 chat message dict."""
    return {"role": role, "content": content}


def static_reset():
    global _static_monitor, _static_episode, _static_chat
    _static_monitor = BenchmarkMonitor(target_win_rate=0.70, min_confusion_delta=4.0, kill_threshold=0.50, window=5)
    _static_episode = {"idx": 0, "step": 0, "confusion": 10.0, "attention": 0.1, "domain": "procedural", "killed": False}
    prob = SCRIPTED_PROBLEMS[0]
    student_msg = (
        f"❓ Problem 1: {prob['question']}\n"
        f"[Student state: Confusion={prob['confusion']}, Attention={prob['attention']} — Impossible mode]"
    )
    _static_chat = [_msg("assistant", student_msg)]
    meter = _build_meter_html(1.0, "OK")
    stats = "Episodes: 0 | Wins: 0 | Win Rate: —"
    return _static_chat, meter, stats, gr.update(interactive=True)


def static_step():
    global _static_episode, _static_chat

    ep = _static_episode
    if ep["killed"]:
        return _static_chat, _build_meter_html(_static_monitor.rolling_win_rate, "RESTARTING"), "Agent restarted.", gr.update(interactive=False)

    prob = SCRIPTED_PROBLEMS[ep["idx"] % len(SCRIPTED_PROBLEMS)]
    confusion = ep["confusion"]
    attention = ep["attention"]
    domain    = prob["domain"]

    # Strategy classification
    strategy = _classify_strategy(confusion, attention, domain, None)
    tutor_msg = _tutor_message(strategy, domain, confusion)

    # Scripted failure: impossible student — confusion barely moves
    confusion_new = max(8.5, confusion - random.uniform(0.0, 0.4))
    attention_new = max(0.0, attention + random.uniform(-0.3, 0.1))

    student_reply = _student_reply(confusion_new, attention_new)

    _static_chat.append(_msg("user", f"[{strategy.upper()}] {tutor_msg}"))
    _static_chat.append(_msg("assistant", student_reply))
    ep["step"] += 1
    ep["confusion"] = confusion_new
    ep["attention"]  = attention_new

    # End episode after 5 steps (scripted timeout)
    if ep["step"] >= 5:
        outcome = "success" if confusion_new <= 2.0 else "timeout"
        signal = _static_monitor.record_episode(
            outcome=outcome,
            confusion_start=prob["confusion"],
            confusion_end=confusion_new,
            steps=ep["step"],
        )

        ep["idx"]  += 1
        ep["step"]  = 0
        ep["killed"] = signal == WatchdogSignal.KILL_AGENT

        if signal == WatchdogSignal.KILL_AGENT:
            _static_chat.append(_msg("assistant", "💀 [WATCHDOG] AGENT PERFORMANCE CRITICAL — KILL SIGNAL SENT"))
            _static_chat.append(_msg("assistant", "🔄 [WATCHDOG] Re-initializing agent from baseline weights..."))
            _static_chat.append(_msg("assistant", "✅ [WATCHDOG] Agent restarted. New session initiated."))
            _static_monitor.acknowledge_restart()
            ep["killed"] = False

        # Load next problem
        if ep["idx"] < len(SCRIPTED_PROBLEMS):
            next_prob = SCRIPTED_PROBLEMS[ep["idx"]]
            ep["confusion"] = next_prob["confusion"]
            ep["attention"]  = next_prob["attention"]
            ep["domain"]     = next_prob["domain"]
            _static_chat.append(_msg(
                "assistant",
                f"❓ Problem {ep['idx']+1}: {next_prob['question']}\n"
                f"[Confusion={next_prob['confusion']}, Attention={next_prob['attention']}]"
            ))

    wr = _static_monitor.rolling_win_rate
    sig = _static_monitor.signal.value
    meter = _build_meter_html(wr, sig)
    s = _static_monitor.status_dict()
    stats = f"Episodes: {s['total_episodes']} | Wins: {s['total_wins']} | Rolling Win Rate: {int(wr*100)}% | Kills: {s['kill_count']}"
    interactive = ep["idx"] < len(SCRIPTED_PROBLEMS)
    return _static_chat, meter, stats, gr.update(interactive=interactive)


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC MODE  — Human-in-the-loop Sandbox
# ═══════════════════════════════════════════════════════════════════════════════

_dyn_monitor = BenchmarkMonitor(target_win_rate=0.70, min_confusion_delta=4.0, kill_threshold=0.40, window=5)
_dyn_state = {"confusion": 7.5, "attention": 6.0, "domain": "factual", "step": 0, "ep": 0, "sat_history": deque(maxlen=5), "last_strategy": None}
_dyn_chat: list[dict] = []
_dyn_mode_locked = False  # fail-safe flag


def dyn_reset():
    global _dyn_monitor, _dyn_state, _dyn_chat, _dyn_mode_locked
    _dyn_monitor   = BenchmarkMonitor(target_win_rate=0.70, min_confusion_delta=4.0, kill_threshold=0.40, window=5)
    _dyn_mode_locked = False
    domain = random.choice(["factual", "procedural", "conceptual", "transfer"])
    _dyn_state = {"confusion": random.uniform(6.0, 8.5), "attention": random.uniform(4.0, 7.0),
                  "domain": domain, "step": 0, "ep": 0, "sat_history": deque(maxlen=5), "last_strategy": None}
    _dyn_chat = [_msg("assistant",
        f"👋 New session! Domain: {domain.upper()}\n"
        f"Confusion={_dyn_state['confusion']:.1f} | Attention={_dyn_state['attention']:.1f}\n"
        "Reply as the student and rate your satisfaction below."
    )]
    return _dyn_chat, _build_meter_html(1.0, "OK"), "Awaiting first interaction...", gr.update(interactive=True), 3


def dyn_step(user_input: str, satisfaction: int):
    global _dyn_chat, _dyn_state, _dyn_mode_locked

    if _dyn_mode_locked:
        _dyn_chat.append(_msg("assistant", "⚠️ Dynamic mode locked by fail-safe. Switch to Static tab or reset."))
        return _dyn_chat, _build_meter_html(_dyn_monitor.rolling_win_rate, "WARN"), "Locked.", gr.update(interactive=False), satisfaction

    if not user_input.strip():
        return _dyn_chat, _build_meter_html(_dyn_monitor.rolling_win_rate, _dyn_monitor.signal.value), "", gr.update(interactive=True), satisfaction

    st = _dyn_state
    _dyn_chat.append(_msg("user", user_input))

    strategy = _classify_strategy(st["confusion"], st["attention"], st["domain"], st["last_strategy"])
    st["last_strategy"] = strategy

    sat_weight = satisfaction / 5.0
    conf_drop  = random.uniform(0.5, 1.8) * sat_weight
    att_change = random.uniform(-0.3, 0.5) * sat_weight

    conf_new = max(0.0, st["confusion"] - conf_drop)
    att_new  = max(0.0, min(10.0, st["attention"] + att_change))

    tutor_reply = (
        f"[{strategy.upper()}] {_tutor_message(strategy, st['domain'], conf_new)}\n"
        f"Confusion: {conf_new:.2f} (dropped {conf_drop:.2f}) | Attention: {att_new:.2f}"
    )
    _dyn_chat.append(_msg("assistant", tutor_reply))

    st["sat_history"].append(satisfaction)
    st["confusion"] = conf_new
    st["attention"]  = att_new
    st["step"]      += 1

    done = conf_new <= 2.0 or st["step"] >= 8
    if done:
        outcome = "success" if conf_new <= 2.0 else "timeout"
        signal = _dyn_monitor.record_episode(outcome=outcome, confusion_start=7.5, confusion_end=conf_new, steps=st["step"])
        st["ep"] += 1
        st["step"] = 0
        st["confusion"] = random.uniform(6.0, 8.5)
        st["attention"]  = random.uniform(4.0, 7.0)
        st["domain"]    = random.choice(["factual", "procedural", "conceptual", "transfer"])

        if signal == WatchdogSignal.KILL_AGENT:
            _dyn_chat.append(_msg("assistant", "💀 [WATCHDOG] Dynamic agent killed — performance below 40%. Restarting..."))
            _dyn_monitor.acknowledge_restart()
            _dyn_chat.append(_msg("assistant", "✅ Agent restarted. New episode begins."))

        _dyn_chat.append(_msg("assistant",
            f"🔄 New episode! Domain: {st['domain'].upper()} | Confusion: {st['confusion']:.1f}"
        ))

    wr  = _dyn_monitor.rolling_win_rate
    sig = _dyn_monitor.signal.value
    s   = _dyn_monitor.status_dict()
    stats = f"Episodes: {s['total_episodes']} | Win Rate: {int(wr*100)}% | Kills: {s['kill_count']}"
    return _dyn_chat, _build_meter_html(wr, sig), stats, gr.update(interactive=True), 3


def toggle_failsafe():
    global _dyn_mode_locked
    _dyn_mode_locked = not _dyn_mode_locked
    label = "🔒 FAIL-SAFE: ON (Dynamic locked)" if _dyn_mode_locked else "🔓 FAIL-SAFE: OFF"
    return label


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="EduForge — RL Tutor Watchdog Demo") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px">
      <span style="font-size:2em;letter-spacing:0.18em;color:#5bc4f5;font-weight:900">EDUFORGE</span>
      <span style="font-size:0.8em;color:#3a5060;display:block;letter-spacing:0.2em;margin-top:2px">
        SELF-IMPROVING RL TUTOR · BENCHMARK WATCHDOG DEMO
      </span>
    </div>
    """)

    with gr.Tabs():

        # ── TAB 1: STATIC ────────────────────────────────────────────────────
        with gr.Tab("📋 Static Demo — Kill/Restart"):

            gr.HTML("""<div style="color:#8899cc;font-size:12px;padding:6px 0 14px">
            Scripted failure mode: 5 "Impossible" student problems. Watch the benchmark meter drop into the
            <span style="color:#cf4a2a">RED ZONE</span> and trigger the Kill/Restart sequence.
            </div>""")

            with gr.Row():
                with gr.Column(scale=2):
                    static_chatbot = gr.Chatbot(label="Tutor ↔ Student Session", height=420)
                with gr.Column(scale=1):
                    static_meter   = gr.HTML(label="Live Benchmark")
                    static_stats   = gr.Textbox(label="Session Stats", interactive=False, lines=2)

            with gr.Row():
                static_reset_btn = gr.Button("⟳ Reset Demo", variant="secondary")
                static_step_btn  = gr.Button("▶ Next Tutor Step", variant="primary")

            static_reset_btn.click(static_reset, outputs=[static_chatbot, static_meter, static_stats, static_step_btn])
            static_step_btn.click(static_step,  outputs=[static_chatbot, static_meter, static_stats, static_step_btn])

            gr.HTML("""<div style="font-size:10px;color:#3a5060;padding-top:8px">
            Strategy classification: Analogy · Socratic · Worked Example · Fact Correction · Explanation · Hint
            </div>""")

        # ── TAB 2: DYNAMIC ───────────────────────────────────────────────────
        with gr.Tab("🧪 Dynamic Sandbox — Human-in-the-Loop"):

            gr.HTML("""<div style="color:#8899cc;font-size:12px;padding:6px 0 14px">
            You are the student. Reply to the tutor. Rate your satisfaction — low scores drop the benchmark
            and may trigger the Kill/Restart watchdog in real time.
            </div>""")

            with gr.Row():
                with gr.Column(scale=2):
                    dyn_chatbot = gr.Chatbot(label="Live Session", height=380)
                    dyn_input   = gr.Textbox(placeholder="Type your student response…", label="Your reply", lines=2)
                    dyn_sat     = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="😐 Satisfaction (1=Terrible · 5=Great)")
                    with gr.Row():
                        dyn_send_btn  = gr.Button("Send →", variant="primary")
                        dyn_reset_btn = gr.Button("⟳ New Session", variant="secondary")
                        dyn_failsafe  = gr.Button("🔓 FAIL-SAFE: OFF", variant="secondary")

                with gr.Column(scale=1):
                    dyn_meter = gr.HTML(label="Live Benchmark")
                    dyn_stats = gr.Textbox(label="Session Stats", interactive=False, lines=2)

            dyn_reset_btn.click(dyn_reset,    outputs=[dyn_chatbot, dyn_meter, dyn_stats, dyn_send_btn, dyn_sat])
            dyn_send_btn.click(dyn_step,      inputs=[dyn_input, dyn_sat],
                               outputs=[dyn_chatbot, dyn_meter, dyn_stats, dyn_send_btn, dyn_sat])
            dyn_input.submit(dyn_step,         inputs=[dyn_input, dyn_sat],
                             outputs=[dyn_chatbot, dyn_meter, dyn_stats, dyn_send_btn, dyn_sat])
            dyn_failsafe.click(toggle_failsafe, outputs=[dyn_failsafe])

    gr.HTML("""<div style="text-align:center;color:#1a2a36;font-size:10px;padding:12px 0 4px;letter-spacing:0.1em">
    OPENENV · PARTIAL OBSERVABILITY · DQN MULTI-HEAD POLICY · © EDUFORGE 2025
    </div>""")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, css=CSS)
