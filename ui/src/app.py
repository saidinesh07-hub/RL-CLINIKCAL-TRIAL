import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request

# IMPORT your RL function
from inference import run_simulation
from env import ClinicalTrialEnv

# 🔥 FastAPI (for OpenEnv)
api = FastAPI()

# 🔥 Initialize environment
env = ClinicalTrialEnv("medium")


# ================= API ENDPOINTS =================

@api.post("/reset")
def reset_api():
    try:
        result = env.reset()
        return {
            "observation": result.get("observation", {})
        }
    except:
        return {
            "observation": {}
        }


@api.post("/step")
async def step_api(request: Request = None):
    action = 0  # default action

    try:
        if request:
            data = await request.json()
            action = data.get("action", 0)
    except:
        action = 0

    result = env.step(action)

    return {
        "observation": result.get("observation", {}),
        "reward": float(result.get("reward", 0.0)),
        "terminated": bool(result.get("terminated", False)),
        "truncated": False,
        "info": {}
    }


# ================= UI FUNCTIONS =================

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


# ================= GRADIO UI =================

with gr.Blocks(title="AI Clinical Trial Optimization") as demo:

    gr.Markdown("## AI Clinical Trial Optimization")

    with gr.Row():
        task_input = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="medium",
            label="Task Difficulty"
        )
        agent_input = gr.Dropdown(
            choices=["random", "rule_based", "greedy_fairness", "q_learning"],
            value="rule_based",
            label="Agent Type"
        )
        episodes_input = gr.Slider(
            minimum=1, maximum=50, value=10, step=1,
            label="Number of Episodes"
        )

    btn = gr.Button("Run Simulation")

    summary_out = gr.Textbox(label="Summary")
    table_out = gr.Dataframe(label="Episode Results")
    plot_out = gr.Plot(label="Reward vs Episode")


    def on_run_simulation(task, agent, episodes):
        score, episodes_data = run_simulation(task, agent, episodes)

        summary = f"Final Score: {score:.4f}"
        table = create_table(episodes_data)
        plot = create_plot(episodes_data)

        return summary, table, plot


    btn.click(
        on_run_simulation,
        inputs=[task_input, agent_input, episodes_input],
        outputs=[summary_out, table_out, plot_out]
    )


# ================= LAUNCH =================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)