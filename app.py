import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from env import ClinicalTrialEnv


env = ClinicalTrialEnv("medium")
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
        "green": "badge-green",
        "yellow": "badge-yellow",
        "red": "badge-red",
        "blue": "badge-blue",
        "indigo": "badge-indigo",
        "orange": "badge-orange",
    }
    return f'<span class="badge {tone_map.get(tone, tone_map["neutral"])}">{esc(text)}</span>'


def status_html(message: str, kind: str = "info"):
    color_map = {
        "success": "#16a34a",
        "error": "#dc2626",
        "info": "#2563eb",
        "warning": "#f59e0b",
    }
    icon_map = {
        "success": "✅",
        "error": "❌",
        "info": "ℹ️",
        "warning": "⚠️",
    }
    color = color_map.get(kind, color_map["info"])
    icon = icon_map.get(kind, icon_map["info"])
    return (
        f'<div class="status-box" style="border-color:{color};color:{color};">'
        f'<span class="status-icon">{icon}</span>{esc(message)}</div>'
    )


def patient_card(title: str, icon: str, value: str, tone: str = "blue"):
    return f'''
        <div class="mini-card {tone}">
            <div class="mini-card-title">{icon} {esc(title)}</div>
            <div class="mini-card-value">{value}</div>
        </div>
    '''


def patient_section(state):
    patient = state.get("patient", {}) or {}
    comorbidities = patient.get("comorbidities", []) or []
    tags = "".join(badge(item, "indigo") for item in comorbidities) if comorbidities else badge("None", "neutral")
    return {
        "patient_id": patient_card("Patient", "🧑", esc(patient.get("patient_id")), "blue"),
        "condition": patient_card("Condition", "❤️", esc(patient.get("condition")), "indigo"),
        "severity": patient_card("Severity", "📊", esc(patient.get("severity")), "orange"),
        "age_group": patient_card("Age Group", "👥", esc(patient.get("age_group")), "blue"),
        "gender": patient_card("Gender", "⚧", esc(patient.get("gender")), "indigo"),
        "comorbidities": f'''
            <div class="mini-card full-width">
                <div class="mini-card-title">🏷️ Comorbidities</div>
                <div class="tag-row">{tags}</div>
            </div>
        ''',
    }


def summary_stats(state):
    system = state.get("system", {}) or {}
    total = max(int(system.get("total_patients", 1)), 1)
    assigned = int(system.get("patients_assigned", 0))
    rejected = int(system.get("patients_rejected", 0))
    fill_ratio = assigned / total
    return f'''
        <div class="top-stats">
            <div class="stat-pill"><span>Step</span><strong>{esc(system.get("step", 0))}</strong></div>
            <div class="stat-pill"><span>Assigned</span><strong>{assigned}</strong></div>
            <div class="stat-pill"><span>Rejected</span><strong>{rejected}</strong></div>
            <div class="stat-pill"><span>Throughput</span><strong>{fill_ratio:.0%}</strong></div>
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


def bar_indicator(label: str, value: int, total: int, accent: str = "blue"):
    total = max(int(total), 1)
    ratio = max(0.0, min(float(value) / total, 1.0))
    return f'''
        <div class="progress-wrap">
            <div class="progress-head"><span>{esc(label)}</span><strong>{value}/{total}</strong></div>
            <div class="progress-track">
                <div class="progress-fill {accent}" style="width:{ratio * 100:.1f}%;"></div>
            </div>
        </div>
    '''


def html_table(state):
    trials = state.get("trials", [])
    rows = []
    for trial in trials:
        slots = int(trial.get("slots_remaining", 0))
        fill_rate = float(trial.get("fill_rate", 0.0))
        priority = int(trial.get("priority", 0))
        if slots < 3:
            row_class = "row-red"
        elif slots < 6:
            row_class = "row-yellow"
        else:
            row_class = "row-default"
        priority_tag = badge(priority, "orange" if priority >= 3 else "yellow" if priority == 2 else "neutral")
        if priority >= 3:
            priority_tag = badge(f"High {priority}", "orange")
        rows.append(
            f'''
                <tr class="{row_class}">
                    <td>{esc(trial.get("trial_id", ""))}</td>
                    <td>{esc(trial.get("condition", ""))}</td>
                    <td>{slots}</td>
                    <td>{priority_tag}</td>
                    <td>{fill_rate:.2f}</td>
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
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows) if rows else '<tr><td colspan="5">No trials available</td></tr>'}
                </tbody>
            </table>
        </div>
    '''


def reward_plot():
    figure = plt.figure(figsize=(5.8, 2.8), facecolor="#0f172a")
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
            axis.plot(dense_x, dense_y, color="#22c55e", linewidth=2.8, alpha=0.9)
        axis.plot(x_values, y_values, color="#60a5fa", linewidth=0, marker="o", markersize=5)
        axis.set_title("Reward Progression", color="#e2e8f0", pad=10, fontsize=11)
        axis.set_xlabel("Step", color="#cbd5e1")
        axis.set_ylabel("Reward", color="#cbd5e1")
    else:
        axis.set_title("Reward Progression", color="#e2e8f0", pad=10, fontsize=11)
        axis.text(0.5, 0.5, "No reward data yet", ha="center", va="center", color="#94a3b8", fontsize=11, transform=axis.transAxes)
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    return figure


def trials_table_rows(state):
    rows = []
    for trial in state.get("trials", []):
        rows.append([
            trial.get("trial_id", ""),
            trial.get("condition", ""),
            trial.get("slots_remaining", 0),
            trial.get("priority", 0),
            f'{float(trial.get("fill_rate", 0.0)):.2f}',
        ])
    return rows


def render_dashboard(state, status_message, kind, slider_max):
    patient_html = patient_section(state)
    system = state.get("system", {}) or {}
    assigned = int(system.get("patients_assigned", 0))
    rejected = int(system.get("patients_rejected", 0))
    total = max(int(system.get("total_patients", 1)), 1)
    diversity_index = float(system.get("diversity_index", 0.0))
    step_num = fmt_value(system.get("step", 0))
    return (
        patient_html["patient_id"],
        patient_html["condition"],
        patient_html["severity"],
        patient_html["age_group"],
        patient_html["gender"],
        patient_html["comorbidities"],
        html_table(state),
        step_num,
        bar_indicator("Assigned Patients", assigned, total, "blue"),
        bar_indicator("Rejected Patients", rejected, total, "orange"),
        progress_bar("Diversity Index", diversity_index, "indigo"),
        reward_plot(),
        status_html(status_message, kind),
        gr.update(maximum=slider_max, value=0, interactive=not terminated_flag),
        f'<div class="action-chip">Selected Action: <strong>0</strong></div>',
        gr.update(interactive=not terminated_flag),
        summary_stats(state),
    )


def reset_simulation(difficulty: str):
    global env, terminated_flag, reward_history
    env = ClinicalTrialEnv(difficulty)
    terminated_flag = False
    reward_history = []
    env.reset()
    return render_dashboard(
        env.state(),
        f"Simulation reset for {difficulty.capitalize()} task.",
        "success",
        env.n_trials,
    )


def step_simulation(action: int):
    global terminated_flag, reward_history

    if terminated_flag:
        return render_dashboard(
            env.state(),
            "Episode already terminated. Click Reset Simulation.",
            "error",
            env.n_trials,
        )

    try:
        safe_action = int(max(0, min(int(action), env.n_trials)))
        result = env.step(safe_action)
        terminated_flag = bool(result.get("terminated", False))
        reward_history.append(float(result.get("reward", 0.0)))
        message = "Episode terminated." if terminated_flag else f"Step completed with action {safe_action}."
        kind = "success" if not terminated_flag else "info"
        return render_dashboard(
            result.get("observation", env.state()),
            message,
            kind,
            env.n_trials,
        )
    except AssertionError:
        terminated_flag = True
        return render_dashboard(
            env.state(),
            "Episode already terminated. Click Reset Simulation.",
            "error",
            env.n_trials,
        )


def action_changed(action: int):
    return f'<div class="action-chip">Selected Action: <strong>{int(action)}</strong></div>'



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

        .gradio-container { background:
            radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 35%),
            radial-gradient(circle at top right, rgba(99,102,241,0.15), transparent 30%),
            linear-gradient(180deg, #050b14 0%, #0a1220 100%) !important;
            color: var(--text) !important;
        }

        .dashboard-shell { max-width: 1380px; margin: 0 auto; padding: 14px 4px 26px; }
        .hero { padding: 10px 4px 18px; }
        .hero h1 { font-size: 2.35rem; line-height: 1.1; margin-bottom: 8px; letter-spacing: -0.04em; }
        .hero-subtitle { color: var(--muted); font-size: 1.03rem; margin-bottom: 14px; }

        .top-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
        .stat-pill { background: linear-gradient(180deg, rgba(18, 27, 46, 0.98), rgba(9, 15, 27, 0.94)); border: 1px solid var(--border); border-radius: 18px; padding: 14px 16px; box-shadow: var(--shadow); }
        .stat-pill span { display: block; color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px; }
        .stat-pill strong { font-size: 1.25rem; color: #ffffff; }

        .section-card { position: relative; overflow: hidden; border: 1px solid var(--border); border-radius: 24px; padding: 18px; background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(10, 16, 28, 0.96)); box-shadow: var(--shadow); backdrop-filter: blur(18px); }
        .section-card::before { content: ''; position: absolute; inset: 0; border-radius: 24px; padding: 1px; background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(99,102,241,0.08), rgba(34,197,94,0.08)); -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0); -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none; }
        .section-title { margin: 0 0 12px 0; }

        .mini-card { min-height: 88px; border-radius: 18px; border: 1px solid rgba(148,163,184,.16); padding: 14px 16px; background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015)); box-shadow: inset 0 1px 0 rgba(255,255,255,0.03); }
        .mini-card.full-width { width: 100%; }
        .mini-card-title { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
        .mini-card-value { color: #f8fafc; font-size: 1.02rem; font-weight: 600; line-height: 1.4; }
        .tag-row { display: flex; flex-wrap: wrap; gap: 8px; }

        .badge { display: inline-flex; align-items: center; border-radius: 999px; padding: 6px 10px; font-size: 0.8rem; font-weight: 700; border: 1px solid transparent; }
        .badge-neutral { background: rgba(148,163,184,.14); color: #cbd5e1; border-color: rgba(148,163,184,.16); }
        .badge-blue { background: rgba(59,130,246,.14); color: #bfdbfe; border-color: rgba(59,130,246,.22); }
        .badge-indigo { background: rgba(99,102,241,.14); color: #c7d2fe; border-color: rgba(99,102,241,.22); }
        .badge-green { background: rgba(34,197,94,.14); color: #bbf7d0; border-color: rgba(34,197,94,.22); }
        .badge-yellow { background: rgba(245,158,11,.14); color: #fde68a; border-color: rgba(245,158,11,.24); }
        .badge-red { background: rgba(239,68,68,.14); color: #fecaca; border-color: rgba(239,68,68,.24); }
        .badge-orange { background: rgba(249,115,22,.14); color: #fed7aa; border-color: rgba(249,115,22,.24); }

        .table-shell { overflow: hidden; border-radius: 18px; border: 1px solid rgba(148,163,184,.14); }
        .trial-table { width: 100%; border-collapse: collapse; background: rgba(8, 13, 24, .75); }
        .trial-table thead th { text-align: left; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #cbd5e1; background: rgba(16, 23, 40, 0.98); padding: 14px 16px; }
        .trial-table tbody td { padding: 14px 16px; border-top: 1px solid rgba(148,163,184,.08); color: #e2e8f0; }
        .trial-table tbody tr { transition: transform .18s ease, background .18s ease; }
        .trial-table tbody tr:hover { transform: translateY(-1px); background: rgba(255,255,255,.03); }
        .row-red { background: rgba(127, 29, 29, 0.22); }
        .row-yellow { background: rgba(120, 53, 15, 0.18); }
        .row-default { background: transparent; }

        .progress-wrap { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
        .progress-head { display: flex; justify-content: space-between; color: #e2e8f0; font-size: 0.92rem; }
        .progress-track { width: 100%; height: 12px; border-radius: 999px; background: rgba(148,163,184,.16); overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 999px; box-shadow: 0 0 24px rgba(59,130,246,.18); transition: width .35s ease; }
        .progress-fill.blue { background: linear-gradient(90deg, #2563eb, #38bdf8); }
        .progress-fill.indigo { background: linear-gradient(90deg, #4f46e5, #818cf8); }
        .progress-fill.green { background: linear-gradient(90deg, #16a34a, #4ade80); }
        .progress-fill.orange { background: linear-gradient(90deg, #ea580c, #fb923c); }

        .status-box { display: flex; align-items: center; gap: 10px; padding: 12px 16px; border-radius: 16px; border: 1px solid; background: rgba(255,255,255,0.03); font-weight: 700; box-shadow: 0 10px 30px rgba(0,0,0,.16); }
        .status-icon { font-size: 1.05rem; }

        .action-chip { display: inline-flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 999px; background: rgba(59,130,246,.10); border: 1px solid rgba(59,130,246,.18); color: #dbeafe; font-weight: 700; margin-bottom: 12px; }

        .gr-button { min-height: 54px !important; border-radius: 16px !important; font-weight: 800 !important; letter-spacing: 0.01em; }
        .gr-button:hover { transform: translateY(-1px); box-shadow: 0 14px 34px rgba(59,130,246,.22); }

        .info-note { color: var(--muted); font-size: 0.9rem; margin-top: 8px; }
        .action-stack { display: grid; gap: 12px; }

        @media (max-width: 900px) {
            .top-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 640px) {
            .top-stats { grid-template-columns: 1fr; }
            .hero h1 { font-size: 1.75rem; }
        }
    """,
) as demo:
    with gr.Column(elem_classes=["dashboard-shell"]):
        with gr.Column(elem_classes=["hero"]):
            gr.Markdown("# ⚡ RL Clinical Trial Optimization")
            gr.Markdown("AI-powered patient-trial matching system", elem_classes=["hero-subtitle"])
            stats_summary = gr.HTML()

        with gr.Row():
            difficulty = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="medium",
                label="Task Difficulty",
                info="Select the environment difficulty for the simulation.",
            )

        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("## 👨‍⚕️ Current Patient", elem_classes=["section-title"])
            with gr.Row():
                patient_id = gr.HTML()
                condition = gr.HTML()
                severity = gr.HTML()
            with gr.Row():
                age_group = gr.HTML()
                gender = gr.HTML()
                comorbidities = gr.HTML()

        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("## 📊 Trials Table", elem_classes=["section-title"])
            trials_table = gr.HTML()

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["section-card"]):
                gr.Markdown("## 📈 System Status", elem_classes=["section-title"])
                step_num = gr.Textbox(label="Step", interactive=False)
                assigned = gr.HTML()
                rejected = gr.HTML()
                diversity = gr.HTML()

            with gr.Column(scale=1, elem_classes=["section-card"]):
                gr.Markdown("## Reward Trend", elem_classes=["section-title"])
                reward_chart = gr.Plot(label="Reward Progression")

        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("## ⚙️ Action Panel", elem_classes=["section-title"])
            action_value = gr.HTML()
            action = gr.Slider(minimum=0, maximum=8, step=1, value=0, label="Action", info="0 = Reject, 1–N = Assign to trial")
            with gr.Row():
                step_btn = gr.Button("Step", variant="primary")
                reset_btn = gr.Button("Reset Simulation", variant="secondary")
            gr.Markdown("0 = Reject, 1–N = Assign to trial", elem_classes=["info-note"])

        status = gr.HTML()

    outputs = [
        patient_id,
        condition,
        severity,
        age_group,
        gender,
        comorbidities,
        trials_table,
        step_num,
        assigned,
        rejected,
        diversity,
        reward_chart,
        status,
        action,
        action_value,
        step_btn,
        stats_summary,
    ]

    reset_btn.click(fn=reset_simulation, inputs=[difficulty], outputs=outputs)
    step_btn.click(fn=step_simulation, inputs=[action], outputs=outputs)
    action.change(fn=action_changed, inputs=[action], outputs=[action_value])
    demo.load(fn=reset_simulation, inputs=[difficulty], outputs=outputs)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)