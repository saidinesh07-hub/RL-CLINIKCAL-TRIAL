import matplotlib.pyplot as plt
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


def status_html(message: str, kind: str = "info"):
    color_map = {
        "success": "#16a34a",
        "error": "#dc2626",
        "info": "#2563eb",
    }
    color = color_map.get(kind, color_map["info"])
    return f'<div style="padding:10px 14px;border-radius:12px;border:1px solid {color};background:rgba(255,255,255,0.04);color:{color};font-weight:600;">{message}</div>'


def reward_plot():
    figure = plt.figure(figsize=(5.8, 2.8), facecolor="#0f172a")
    axis = figure.add_subplot(111)
    axis.set_facecolor("#0f172a")
    axis.tick_params(colors="#cbd5e1")
    for spine in axis.spines.values():
        spine.set_color("#334155")
    axis.grid(True, color="#334155", alpha=0.35, linewidth=0.8)

    if reward_history:
        axis.plot(range(1, len(reward_history) + 1), reward_history, color="#22c55e", linewidth=2.5, marker="o", markersize=4)
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


def patient_fields(state):
    patient = state.get("patient", {}) or {}
    return (
        fmt_value(patient.get("patient_id")),
        fmt_value(patient.get("condition")),
        fmt_value(patient.get("severity")),
        fmt_value(patient.get("age_group")),
        fmt_value(patient.get("gender")),
        fmt_value(patient.get("comorbidities")),
    )


def system_fields(state):
    system = state.get("system", {}) or {}
    return (
        fmt_value(system.get("step", 0)),
        fmt_value(system.get("patients_assigned", 0)),
        fmt_value(system.get("patients_rejected", 0)),
        f'{float(system.get("diversity_index", 0.0)):.2f}',
    )


def render_dashboard(state, status_message, kind, slider_max):
    patient_id, condition, severity, age_group, gender, comorbidities = patient_fields(state)
    step_num, assigned, rejected, diversity = system_fields(state)
    return (
        patient_id,
        condition,
        severity,
        age_group,
        gender,
        comorbidities,
        trials_table_rows(state),
        step_num,
        assigned,
        rejected,
        diversity,
        reward_plot(),
        status_html(status_message, kind),
        gr.update(maximum=slider_max, value=0),
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


with gr.Blocks(
    title="RL Clinical Trial Optimization",
    css="""
        .dashboard-shell { max-width: 1280px; margin: 0 auto; }
        .section-card { border: 1px solid rgba(148,163,184,.22); border-radius: 18px; padding: 16px; background: rgba(15,23,42,.55); box-shadow: 0 10px 30px rgba(0,0,0,.18); }
        .section-title { margin: 0 0 10px 0; }
        .field-label { font-size: 0.82rem; color: #94a3b8; margin-bottom: 4px; letter-spacing: .02em; text-transform: uppercase; }
    """,
) as demo:
    with gr.Column(elem_classes=["dashboard-shell"]):
        gr.Markdown("# ⚡ RL Clinical Trial Optimization")
        gr.Markdown("Clean live dashboard for patient-trial matching, system status, and step control.")

        with gr.Row():
            difficulty = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="medium",
                label="Task Difficulty",
            )

        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("## 👨‍⚕️ Current Patient", elem_classes=["section-title"])
            with gr.Row():
                patient_id = gr.Textbox(label="Patient ID", interactive=False)
                condition = gr.Textbox(label="Condition", interactive=False)
                severity = gr.Textbox(label="Severity", interactive=False)
            with gr.Row():
                age_group = gr.Textbox(label="Age Group", interactive=False)
                gender = gr.Textbox(label="Gender", interactive=False)
                comorbidities = gr.Textbox(label="Comorbidities", interactive=False)

        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("## 📊 Trials Table", elem_classes=["section-title"])
            trials_table = gr.Dataframe(
                headers=["Trial ID", "Condition", "Slots Remaining", "Priority", "Fill Rate"],
                datatype=["number", "str", "number", "number", "str"],
                interactive=False,
                row_count=6,
                col_count=(5, "fixed"),
                wrap=True,
            )

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["section-card"]):
                gr.Markdown("## 📈 System Status", elem_classes=["section-title"])
                step_num = gr.Textbox(label="Step", interactive=False)
                assigned = gr.Textbox(label="Assigned Patients", interactive=False)
                rejected = gr.Textbox(label="Rejected Patients", interactive=False)
                diversity = gr.Textbox(label="Diversity Index", interactive=False)

            with gr.Column(scale=1, elem_classes=["section-card"]):
                gr.Markdown("## Reward Trend", elem_classes=["section-title"])
                reward_chart = gr.Plot(label="Reward Progression")

        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("## ⚙️ Action Panel", elem_classes=["section-title"])
            action = gr.Slider(minimum=0, maximum=8, step=1, value=0, label="Action")
            with gr.Row():
                step_btn = gr.Button("Step", variant="primary")
                reset_btn = gr.Button("Reset Simulation", variant="secondary")

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
    ]

    reset_btn.click(fn=reset_simulation, inputs=[difficulty], outputs=outputs)
    step_btn.click(fn=step_simulation, inputs=[action], outputs=outputs)
    demo.load(fn=reset_simulation, inputs=[difficulty], outputs=outputs)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)