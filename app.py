import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

# ✅ FIXED IMPORT
from inference import run_inference
from env import ClinicalTrialEnv

# Global env
env = ClinicalTrialEnv("medium")


# ================= OPENENV API =================

def reset():
    result = env.reset()
    return {
        "observation": result.get("observation", {})
    }


def step(action: int = 0):
    result = env.step(action)
    return {
        "observation": result.get("observation", {}),
        "reward": float(result.get("reward", 0.0)),
        "terminated": bool(result.get("terminated", False)),
        "truncated": False,
        "info": {}
    }


# ================= UI HELPERS =================

def create_table(data):
    return pd.DataFrame(data)


def create_plot(data):
    df = pd.DataFrame(data)
    plt.figure()
    plt.plot(df["episode"], df["reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.grid(True)
    return plt


# ================= MAIN FUNCTION =================

def on_run_simulation(task, agent, episodes):
    results = run_inference(task, agent, int(episodes))

    summary = f"""Final Score: {results['score']}
Mean Reward: {results['mean_reward']}
Assignment Rate: {results['assignment_rate']}
Diversity Index: {results['diversity_index']}"""

    episodes_data = results["episodes"]

    table = create_table(episodes_data)
    plot = create_plot(episodes_data)

    return summary, table, plot


# ================= UI =================

with gr.Blocks(title="AI Clinical Trial Optimization") as demo:

    gr.Markdown("## AI Clinical Trial Optimization")

    with gr.Row():
        task_input = gr.Dropdown(
            ["easy", "medium", "hard"], value="medium", label="Task"
        )
        agent_input = gr.Dropdown(
            ["random", "rule_based", "greedy_fairness", "q_learning"],
            value="rule_based",
            label="Agent"
        )
        episodes_input = gr.Slider(
            1, 50, value=10, step=1, label="Episodes"
        )

    btn = gr.Button("Run Simulation")

    summary_out = gr.Textbox(label="Summary")
    table_out = gr.Dataframe(label="Episodes")
    plot_out = gr.Plot(label="Reward Graph")

    btn.click(
        on_run_simulation,
        inputs=[task_input, agent_input, episodes_input],
        outputs=[summary_out, table_out, plot_out]
    )


# ================= LAUNCH =================

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)