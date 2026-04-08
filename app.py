import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from env import ClinicalTrialEnv


env = ClinicalTrialEnv("medium")
current_state = env.reset()["observation"]
terminated_flag = False
reward_history = []


def fmt_value(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "None"
    if value in (None, ""):
        return "None"
    return str(value)


def esc(value):
    text = fmt_value(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def badge(text: str, tone: str = "neutral"):
    tone_map = {
        "neutral": "badge-neutral",
        "blue": "badge-blue",
        "indigo": "badge-indigo",
        "green": "badge-green",
        "yellow": "badge-yellow",
        "orange": "badge-orange",
        "red": "badge-red",
    }
    return f'<span class="badge {tone_map.get(tone, tone_map["neutral"])}">{esc(text)}</span>'


def status_box(message: str, kind: str = "info"):
    colors = {
        "success": ("#22c55e", "✅"),
        "error": ("#ef4444", "❌"),
        "warning": ("#f59e0b", "⚠️"),
        "info": ("#3b82f6", "ℹ️"),
    }
    color, icon = colors.get(kind, colors["info"])
    return (
        f'<div class="status-box" style="border-color:{color};color:{color};">'
        f'<span class="status-icon">{icon}</span>{esc(message)}</div>'
    )


def mini_card(title: str, icon: str, value: str, accent: str = "blue"):
    return f'''
        <div class="mini-card {accent}">
            <div class="mini-card-title">{icon} {esc(title)}</div>
            <div class="mini-card-value">{value}</div>
        </div>
    '''


def top_summary(state):
    system = state.get("system", {}) or {}
    return f'''
        <div class="top-stats">
            {mini_card("Step", "🧭", esc(system.get("step", 0)), "blue")}
            {mini_card("Assigned", "🟢", esc(system.get("patients_assigned", 0)), "green")}
            {mini_card("Rejected", "🔴", esc(system.get("patients_rejected", 0)), "orange")}
            {mini_card("Acceptance Rate", "📈", f'{float(system.get("acceptance_rate", 0.0)):.2f}', "indigo")}
        </div>
    '''


def patient_section(state):
    patient = state.get("patient", {}) or {}
    comorbidities = patient.get("comorbidities", []) or []
    tags = "".join(badge(item, "indigo") for item in comorbidities) if comorbidities else badge("None", "neutral")

    return f'''
        <div class="patient-grid">
            <div class="mini-card blue">
                <div class="mini-card-title">🧑 Patient ID</div>
                <div class="mini-card-value">{esc(patient.get("patient_id"))}</div>
            </div>
            <div class="mini-card indigo">
                <div class="mini-card-title">❤️ Condition</div>
                <div class="mini-card-value">{esc(patient.get("condition"))}</div>
            </div>
            <div class="mini-card orange">
                <div class="mini-card-title">📊 Severity</div>
                <div class="mini-card-value">{esc(patient.get("severity"))}</div>
            </div>
            <div class="mini-card blue">
                <div class="mini-card-title">👥 Age Group</div>
                <div class="mini-card-value">{esc(patient.get("age_group"))}</div>
            </div>
            <div class="mini-card indigo">
                <div class="mini-card-title">⚧ Gender</div>
                <div class="mini-card-value">{esc(patient.get("gender"))}</div>
            </div>
            <div class="mini-card full-width">
                <div class="mini-card-title">🏷️ Comorbidities</div>
                <div class="tag-row">{tags}</div>
            </div>
        </div>
    '''


def trial_row_class(trial, valid_trial_ids, recommended_action):
    slots = int(trial.get("slots_remaining", 0))
    trial_id = int(trial.get("trial_id", 0))
    classes = ["trial-row"]
    if trial_id == recommended_action:
        classes.append("row-recommended")
    elif trial_id in valid_trial_ids:
        classes.append("row-valid")
    else:
        classes.append("row-invalid")
    if slots < 3:
        classes.append("slots-red")
    elif slots < 6:
        classes.append("slots-yellow")
    return " ".join(classes)


def render_trials_table(state):
    system = state.get("system", {}) or {}
    valid_trial_ids = set(system.get("valid_trials", []))
    recommended_action = int(system.get("recommended_action", 0))
    trial_statuses = {item["trial_id"]: item for item in system.get("trial_statuses", [])}

    rows = []
    for trial in state.get("trials", []):
        trial_id = int(trial.get("trial_id", 0))
        slots = int(trial.get("slots_remaining", 0))
        fill_rate = float(trial.get("fill_rate", 0.0))
        priority = int(trial.get("priority", 0))
        status = trial_statuses.get(trial_id, {})
        valid = bool(status.get("valid", False))
        reason = status.get("reason", "valid" if valid else "invalid")

        if trial_id == recommended_action and recommended_action != 0:
            status_badge = badge("Recommended", "green")
        elif valid:
            status_badge = badge("Valid", "green")
        else:
            status_badge = badge(reason.replace("_", " "), "red")

        priority_badge = badge(f"P{priority}", "orange" if priority >= 3 else "yellow" if priority == 2 else "neutral")
        if priority >= 3:
            priority_badge = badge(f"High {priority}", "orange")

        rows.append(
            f'''
                <tr class="{trial_row_class(trial, valid_trial_ids, recommended_action)}">
                    <td>{trial_id}</td>
                    <td>{esc(trial.get("condition", ""))}</td>
                    <td>{slots}</td>
                    <td>{priority_badge}</td>
                    <td>{fill_rate:.2f}</td>
                    <td>{status_badge}</td>
                </tr>
            '''
        )

    return f'''
        <div class="table-shell">
            <table class="trial-table">
                <thead>
                    <tr>
                        <th>Trial ID</th>
                        <th>Condition</th>
                        <th>Slots Remaining</th>
                        <th>Priority</th>
                        <th>Fill Rate</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows) if rows else '<tr><td colspan="6">No trials available</td></tr>'}
                </tbody>
            </table>
        </div>
    '''


def system_cards(state):
    system = state.get("system", {}) or {}
    average_reward = float(system.get("average_reward", 0.0))
    diversity_index = float(system.get("diversity_index", 0.0))
    invalid_actions = int(system.get("invalid_actions", 0))

    return f'''
        <div class="system-grid">
            <div class="mini-card blue">
                <div class="mini-card-title">🧭 Step</div>
                <div class="mini-card-value">{esc(system.get("step", 0))}</div>
            </div>
            <div class="mini-card green">
                <div class="mini-card-title">🟢 Assigned Patients</div>
                <div class="mini-card-value">{esc(system.get("patients_assigned", 0))}</div>
            </div>
            <div class="mini-card orange">
                <div class="mini-card-title">🔴 Rejected Patients</div>
                <div class="mini-card-value">{esc(system.get("patients_rejected", 0))}</div>
            </div>
            <div class="mini-card indigo">
                <div class="mini-card-title">📈 Acceptance Rate</div>
                <div class="mini-card-value">{float(system.get("acceptance_rate", 0.0)):.2f}</div>
            </div>
            <div class="mini-card full-width">
                <div class="mini-card-title">📊 Average Reward</div>
                <div class="mini-card-value">{average_reward:.2f}</div>
            </div>
            <div class="mini-card full-width">
                <div class="mini-card-title">🛡️ Reliability</div>
                <div class="mini-card-value">Invalid Actions: {invalid_actions}</div>
            </div>
        </div>
        <div class="progress-stack">
            {progress_bar("Diversity Index", diversity_index, "indigo")}
            {progress_bar("Assignment Progress", float(system.get("assignment_rate", 0.0)), "green")}
            {progress_bar("Average Trial Fill", float(system.get("average_fill_rate", 0.0)), "blue")}
        </div>
    '''


def progress_bar(label: str, value: float, accent: str = "indigo"):
    clamped = max(0.0, min(float(value), 1.0))
    return f'''
        <div class="progress-wrap">
            <div class="progress-head"><span>{esc(label)}</span><strong>{clamped:.2f}</strong></div>
            <div class="progress-track">
                <div class="progress-fill {accent}" style="width:{clamped * 100:.1f}%;"></div>
            </div>
        </div>
    '''


def recommendation_box(state):
    system = state.get("system", {}) or {}
    recommended_action = int(system.get("recommended_action", 0))
    reason = system.get("recommended_reason", "")
    valid_trials = system.get("valid_trials", []) or []

    if recommended_action == 0:
        tone = "warning"
        title = "No valid trial available"
        subtitle = esc(reason)
    else:
        tone = "success"
        title = f"Recommended Action: Trial {recommended_action}"
        subtitle = esc(reason)

    valid_preview = ", ".join(str(item) for item in valid_trials) if valid_trials else "None"
    return f'''
        <div class="recommendation-box {tone}">
            <div class="recommendation-title">🎯 {title}</div>
            <div class="recommendation-reason">{subtitle}</div>
            <div class="recommendation-meta">Valid trials: {esc(valid_preview)}</div>
        </div>
    '''


def selected_action_box(state, action):
    system = state.get("system", {}) or {}
    recommended_action = int(system.get("recommended_action", 0))
    valid_trial_ids = set(system.get("valid_trials", []))

    if action == 0:
        tone = "blue"
        message = "Selected action 0 = Reject"
        detail = "Use only when no valid trial fits or rejection is intended."
    elif action in valid_trial_ids:
        if action == recommended_action:
            tone = "green"
            message = f"Selected action {action} matches the recommendation"
            detail = "This is the best-fit trial for the current patient."
        else:
            tone = "indigo"
            message = f"Selected action {action} is valid"
            detail = f"Recommended action is {recommended_action}."
    else:
        tone = "red"
        message = f"Selected action {action} is invalid for this patient"
        detail = "Failure reasons will be shown after stepping."

    return f'''
        <div class="selection-box {tone}">
            <div class="selection-title">🎮 {message}</div>
            <div class="selection-detail">{detail}</div>
            <div class="selection-hint">0 = Reject, 1–N = Assign to trial</div>
        </div>
    '''


def action_hint_box(state, action):
    system = state.get("system", {}) or {}
    recommended_action = int(system.get("recommended_action", 0))
    valid_trial_ids = set(system.get("valid_trials", []))
    if action == recommended_action and action != 0:
        return status_box("Recommended action selected. This is the best-fit trial.", "success")
    if action == 0:
        if valid_trial_ids:
            return status_box("Reject selected. Valid trial options exist, so this may be suboptimal.", "warning")
        return status_box("Reject selected. No valid trial is available.", "info")
    if action in valid_trial_ids:
        return status_box(f"Action {action} is valid. Recommended action is {recommended_action}.", "info")
    return status_box(f"Action {action} is invalid for the current patient.", "error")


def reward_plot():
    figure = plt.figure(figsize=(6.0, 3.0), facecolor="#0f172a")
    axis = figure.add_subplot(111)
    axis.set_facecolor("#0f172a")
    axis.tick_params(colors="#cbd5e1")
    for spine in axis.spines.values():
        spine.set_color("#334155")
    axis.grid(True, color="#334155", alpha=0.35, linewidth=0.8)

    if reward_history:
        x_values = np.arange(1, len(reward_history) + 1)
        y_values = np.array(reward_history, dtype=float)
        if len(x_values) > 2:
            dense_x = np.linspace(x_values.min(), x_values.max(), 200)
            dense_y = np.interp(dense_x, x_values, y_values)
            axis.plot(dense_x, dense_y, color="#60a5fa", linewidth=2.8, alpha=0.9)
        axis.plot(x_values, y_values, color="#22c55e", linewidth=0, marker="o", markersize=5)
        axis.set_title("Reward Progression", color="#e2e8f0", pad=10, fontsize=11)
        axis.set_xlabel("Step", color="#cbd5e1")
        axis.set_ylabel("Reward", color="#cbd5e1")
        axis.set_xlim(1, max(len(reward_history), 2))
        axis.set_ylim(0, 1)
    else:
        axis.set_title("Reward Progression", color="#e2e8f0", pad=10, fontsize=11)
        axis.text(
            0.5,
            0.5,
            "No reward data yet",
            ha="center",
            va="center",
            color="#94a3b8",
            fontsize=11,
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    return figure


def dashboard_state(message: str, kind: str, selected_action: int):
    global current_state
    state = current_state or env.state()
    return (
        top_summary(state),
        patient_section(state),
        render_trials_table(state),
        system_cards(state),
        recommendation_box(state),
        reward_plot(),
        selected_action_box(state, selected_action),
        action_hint_box(state, selected_action),
        status_box(message, kind),
        gr.update(maximum=env.n_trials, value=selected_action, interactive=not terminated_flag),
        gr.update(interactive=not terminated_flag),
    )


def reset_simulation(difficulty: str):
    global env, current_state, terminated_flag, reward_history
    env = ClinicalTrialEnv(difficulty)
    result = env.reset()
    current_state = result["observation"]
    terminated_flag = False
    reward_history = []
    return dashboard_state(
        f"✅ Simulation reset for {difficulty.capitalize()} task.",
        "success",
        0,
    )


def step_simulation(action: int):
    global current_state, terminated_flag

    if terminated_flag:
        return dashboard_state(
            "❌ Episode already terminated. Click Reset Simulation.",
            "error",
            0,
        )

    try:
        safe_action = int(max(0, min(int(action), env.n_trials)))
        result = env.step(safe_action)
        current_state = result.get("observation", env.state())
        terminated_flag = bool(result.get("terminated", False))
        reward_history.append(float(result.get("reward", 0.0)))

        info = result.get("info", {}) or {}
        reason = info.get("reason", "")
        if reason == "assignment_success":
            message = f"✅ Action {safe_action} succeeded."
            kind = "success"
        elif reason == "valid_rejection":
            message = "✅ Rejection accepted because no better trial fit was available."
            kind = "success"
        elif reason == "unjustified_rejection":
            message = "⚠️ Rejected a patient who had valid trial options."
            kind = "warning"
        elif reason == "invalid_trial":
            message = f"❌ Action {safe_action} failed: invalid trial selection."
            kind = "error"
        elif reason == "invalid_action":
            message = "❌ Action failed: invalid action value."
            kind = "error"
        elif "trial_full" in reason:
            message = f"❌ Action {safe_action} failed: trial full."
            kind = "error"
        elif "condition_mismatch" in reason:
            message = f"❌ Action {safe_action} failed: condition mismatch."
            kind = "error"
        elif "severity_mismatch" in reason:
            message = f"❌ Action {safe_action} failed: severity mismatch."
            kind = "error"
        elif "comorbidity_mismatch" in reason:
            message = f"❌ Action {safe_action} failed: comorbidity mismatch."
            kind = "error"
        else:
            message = "✅ Step completed."
            kind = "info"

        if terminated_flag:
            message = f"{message} Episode terminated."
        return dashboard_state(message, kind, 0)
    except Exception as exc:
        current_state = env.state()
        terminated_flag = True
        return dashboard_state(f"❌ Step failed: {esc(exc)}", "error", 0)


def action_changed(action: int):
    state = current_state or env.state()
    return selected_action_box(state, int(action)), action_hint_box(state, int(action))


with gr.Blocks(
    title="RL Clinical Trial Optimization",
    css="""
        :root {
            --bg: #07111f;
            --panel: rgba(10, 16, 28, 0.88);
            --panel-strong: rgba(16, 23, 40, 0.96);
            --border: rgba(148, 163, 184, 0.18);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --blue: #3b82f6;
            --indigo: #6366f1;
            --green: #22c55e;
            --yellow: #f59e0b;
            --red: #ef4444;
            --shadow: 0 22px 60px rgba(0, 0, 0, 0.38);
        }

        .gradio-container {
            background:
                radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 35%),
                radial-gradient(circle at top right, rgba(99,102,241,0.15), transparent 30%),
                linear-gradient(180deg, #050b14 0%, #0a1220 100%) !important;
            color: var(--text) !important;
        }

        .dashboard-shell { max-width: 1400px; margin: 0 auto; padding: 14px 4px 26px; }
        .hero { padding: 10px 4px 18px; }
        .hero h1 { font-size: 2.4rem; line-height: 1.1; margin-bottom: 8px; letter-spacing: -0.04em; }
        .hero-subtitle { color: var(--muted); font-size: 1.03rem; margin-bottom: 16px; }

        .top-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
        .system-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
        .patient-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
        .tag-row { display: flex; flex-wrap: wrap; gap: 8px; }
        .progress-stack { margin-top: 10px; }

        .mini-card {
            min-height: 88px;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,.16);
            padding: 14px 16px;
            background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .mini-card.full-width { grid-column: 1 / -1; }
        .mini-card-title { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
        .mini-card-value { color: #f8fafc; font-size: 1.02rem; font-weight: 600; line-height: 1.4; }
        .mini-card.blue { border-color: rgba(59,130,246,.22); }
        .mini-card.indigo { border-color: rgba(99,102,241,.22); }
        .mini-card.green { border-color: rgba(34,197,94,.22); }
        .mini-card.orange { border-color: rgba(245,158,11,.22); }

        .badge { display: inline-flex; align-items: center; border-radius: 999px; padding: 6px 10px; font-size: 0.8rem; font-weight: 700; border: 1px solid transparent; }
        .badge-neutral { background: rgba(148,163,184,.14); color: #cbd5e1; border-color: rgba(148,163,184,.16); }
        .badge-blue { background: rgba(59,130,246,.14); color: #bfdbfe; border-color: rgba(59,130,246,.22); }
        .badge-indigo { background: rgba(99,102,241,.14); color: #c7d2fe; border-color: rgba(99,102,241,.22); }
        .badge-green { background: rgba(34,197,94,.14); color: #bbf7d0; border-color: rgba(34,197,94,.22); }
        .badge-yellow { background: rgba(245,158,11,.14); color: #fde68a; border-color: rgba(245,158,11,.24); }
        .badge-red { background: rgba(239,68,68,.14); color: #fecaca; border-color: rgba(239,68,68,.24); }
        .badge-orange { background: rgba(249,115,22,.14); color: #fed7aa; border-color: rgba(249,115,22,.24); }

        .section-card {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 18px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(10, 16, 28, 0.96));
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
            margin-bottom: 16px;
        }
        .section-card::before {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 24px;
            padding: 1px;
            background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(99,102,241,0.08), rgba(34,197,94,0.08));
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
        }
        .section-title { margin: 0 0 12px 0; }

        .table-shell { overflow: hidden; border-radius: 18px; border: 1px solid rgba(148,163,184,.14); }
        .trial-table { width: 100%; border-collapse: collapse; background: rgba(8, 13, 24, .75); }
        .trial-table thead th {
            text-align: left;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #cbd5e1;
            background: rgba(16, 23, 40, 0.98);
            padding: 14px 16px;
        }
        .trial-table tbody td { padding: 14px 16px; border-top: 1px solid rgba(148,163,184,.08); color: #e2e8f0; }
        .trial-row { transition: transform .18s ease, background .18s ease; }
        .trial-row:hover { transform: translateY(-1px); }
        .row-valid { background: rgba(16, 185, 129, 0.10); }
        .row-invalid { background: rgba(239, 68, 68, 0.10); }
        .row-recommended { background: rgba(59, 130, 246, 0.14); }
        .slots-red { box-shadow: inset 4px 0 0 rgba(239,68,68,.9); }
        .slots-yellow { box-shadow: inset 4px 0 0 rgba(245,158,11,.9); }

        .progress-wrap { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
        .progress-head { display: flex; justify-content: space-between; color: #e2e8f0; font-size: 0.92rem; }
        .progress-track { width: 100%; height: 12px; border-radius: 999px; background: rgba(148,163,184,.16); overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 999px; box-shadow: 0 0 24px rgba(59,130,246,.18); transition: width .35s ease; }
        .progress-fill.blue { background: linear-gradient(90deg, #2563eb, #38bdf8); }
        .progress-fill.indigo { background: linear-gradient(90deg, #4f46e5, #818cf8); }
        .progress-fill.green { background: linear-gradient(90deg, #16a34a, #4ade80); }

        .recommendation-box,
        .selection-box,
        .status-box {
            border-radius: 18px;
            border: 1px solid var(--border);
            padding: 14px 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,.16);
            background: rgba(255,255,255,0.03);
        }
        .recommendation-box.success, .selection-box.green { border-color: rgba(34,197,94,.45); color: #bbf7d0; }
        .recommendation-box.warning, .selection-box.blue { border-color: rgba(245,158,11,.45); color: #fde68a; }
        .recommendation-box.error, .selection-box.red { border-color: rgba(239,68,68,.45); color: #fecaca; }
        .recommendation-box.indigo, .selection-box.indigo { border-color: rgba(99,102,241,.45); color: #c7d2fe; }
        .recommendation-title, .selection-title { font-weight: 800; font-size: 1rem; margin-bottom: 6px; }
        .recommendation-reason, .selection-detail, .selection-hint, .recommendation-meta { color: #cbd5e1; font-size: 0.92rem; line-height: 1.5; }
        .status-box { display: flex; align-items: center; gap: 10px; font-weight: 800; }
        .status-icon { font-size: 1.05rem; }

        .action-chip { display: inline-flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 999px; background: rgba(59,130,246,.10); border: 1px solid rgba(59,130,246,.18); color: #dbeafe; font-weight: 700; margin-bottom: 12px; }
        .gr-button {
            min-height: 54px !important;
            border-radius: 16px !important;
            font-weight: 800 !important;
            letter-spacing: 0.01em;
        }
        .gr-button:hover { transform: translateY(-1px); box-shadow: 0 14px 34px rgba(59,130,246,.22); }
        .info-note { color: var(--muted); font-size: 0.9rem; margin-top: 8px; }

        @media (max-width: 1100px) {
            .top-stats, .system-grid, .patient-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 640px) {
            .top-stats, .system-grid, .patient-grid { grid-template-columns: 1fr; }
            .hero h1 { font-size: 1.78rem; }
        }
    """,
) as demo:
    with gr.Column(elem_classes=["dashboard-shell"]):
        with gr.Column(elem_classes=["hero"]):
            gr.Markdown("# ⚡ RL Clinical Trial Optimization")
            gr.Markdown("AI-powered patient-trial matching system", elem_classes=["hero-subtitle"])
            summary_stats = gr.HTML()

        with gr.Row():
            difficulty = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="medium",
                label="Task Difficulty",
                info="Select the environment difficulty for the simulation.",
            )

        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("## 👨‍⚕️ Current Patient", elem_classes=["section-title"])
            patient_cards = gr.HTML()

        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("## 📊 Trials Table", elem_classes=["section-title"])
            trials_table = gr.HTML()

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["section-card"]):
                gr.Markdown("## 📈 System Status", elem_classes=["section-title"])
                system_cards_html = gr.HTML()
            with gr.Column(scale=1, elem_classes=["section-card"]):
                gr.Markdown("## Reward Trend", elem_classes=["section-title"])
                reward_chart = gr.Plot(label="Reward Progression")

        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("## 🎯 Recommendation", elem_classes=["section-title"])
            recommendation_html = gr.HTML()

        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("## ⚙️ Action Panel", elem_classes=["section-title"])
            selected_action_html = gr.HTML()
            action_hint_html = gr.HTML()
            action = gr.Slider(
                minimum=0,
                maximum=8,
                step=1,
                value=0,
                label="Action",
                info="0 = Reject, 1–N = Assign to trial",
            )
            with gr.Row():
                step_btn = gr.Button("Step", variant="primary")
                reset_btn = gr.Button("Reset Simulation", variant="secondary")
            gr.Markdown("0 = Reject, 1–N = Assign to trial", elem_classes=["info-note"])

        status = gr.HTML()

    outputs = [
        summary_stats,
        patient_cards,
        trials_table,
        system_cards_html,
        recommendation_html,
        reward_chart,
        selected_action_html,
        action_hint_html,
        status,
        action,
        step_btn,
    ]

    reset_btn.click(fn=reset_simulation, inputs=[difficulty], outputs=outputs)
    step_btn.click(fn=step_simulation, inputs=[action], outputs=outputs)
    action.change(fn=action_changed, inputs=[action], outputs=[selected_action_html, action_hint_html])
    demo.load(fn=reset_simulation, inputs=[difficulty], outputs=outputs)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
