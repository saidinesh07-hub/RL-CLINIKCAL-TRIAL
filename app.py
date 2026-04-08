import gradio as gr
from env import ClinicalTrialEnv


env = ClinicalTrialEnv("medium")
terminated_flag = False


def reset_simulation(difficulty: str):
    global env, terminated_flag
    env = ClinicalTrialEnv(difficulty)
    terminated_flag = False
    result = env.reset()
    return (
        {"observation": result["observation"]},
        "Simulation reset successfully.",
    )


def step_simulation(action: int):
    global terminated_flag

    if terminated_flag:
        return None, "Episode already terminated. Click Reset Simulation."

    try:
        safe_action = int(max(0, min(int(action), env.n_trials)))
        result = env.step(safe_action)
        terminated_flag = bool(result.get("terminated", False))
        status = "Episode terminated." if terminated_flag else "Step completed."
        return {
            "observation": result.get("observation", {}),
            "reward": float(result.get("reward", 0.0)),
            "terminated": terminated_flag,
        }, status
    except AssertionError:
        terminated_flag = True
        return None, "Episode already terminated. Click Reset Simulation."


with gr.Blocks(title="RL Clinical Trial Optimization") as demo:
    gr.Markdown("# RL Clinical Trial Optimization Running ✅")
    gr.Markdown("Run the environment step-by-step and inspect live observations.")

    with gr.Row():
        difficulty = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="medium",
            label="Task Difficulty",
        )
        action = gr.Slider(minimum=0, maximum=8, step=1, value=0, label="Action")

    with gr.Row():
        reset_btn = gr.Button("Reset Simulation", variant="primary")
        step_btn = gr.Button("Step")

    output = gr.JSON(label="Simulation Output")
    status = gr.Textbox(label="Status", interactive=False)

    reset_btn.click(fn=reset_simulation, inputs=[difficulty], outputs=[output, status])
    step_btn.click(fn=step_simulation, inputs=[action], outputs=[output, status])

    demo.load(fn=reset_simulation, inputs=[difficulty], outputs=[output, status])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)