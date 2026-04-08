import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from inference import run_simulation
from env import ClinicalTrialEnv

env = ClinicalTrialEnv("medium")


# 🔥 OpenEnv functions (NO FastAPI)

def reset():
    result = env.reset()
    return {"observation": result.get("observation", {})}


def step(action=0):
    result = env.step(action)
    return {
        "observation": result.get("observation", {}),
        "reward": float(result.get("reward", 0.0)),
        "terminated": bool(result.get("terminated", False)),
        "truncated": False,
        "info": {}
    }


# ================= UI =================

def create_table(data):
    return pd.DataFrame(data)


def create_plot(data):
    df = pd.DataFrame(data)
    plt.figure()
    plt.plot(df["episode"], df["reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    return plt


def on_run_simulation(task, agent, episodes):
    score, episodes_data = run_simulation(task, agent, episodes)

    summary = f"Final Score: {score:.4f}"
    table = create_table(episodes_data)
    plot = create_plot(episodes_data)

    return summary, table, plot


with gr.Blocks(title="AI Clinical Trial Optimization") as demo:

    gr.Markdown("## AI Clinical Trial Optimization")

    with gr.Row():
        task_input = gr.Dropdown(["easy", "medium", "hard"], value="medium")
        agent_input = gr.Dropdown(["random", "rule_based", "greedy_fairness", "q_learning"], value="rule_based")
        episodes_input = gr.Slider(1, 50, value=10, step=1)

    btn = gr.Button("Run Simulation")

    summary_out = gr.Textbox()
    table_out = gr.Dataframe()
    plot_out = gr.Plot()

    btn.click(
        on_run_simulation,
        inputs=[task_input, agent_input, episodes_input],
        outputs=[summary_out, table_out, plot_out]
    )


# 🔥 CRITICAL: expose endpoints via gradio
demo.queue().launch(server_name="0.0.0.0", server_port=7860)