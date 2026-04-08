import gradio as gr
from env import ClinicalTrialEnv
from tasks import TASK_MAP

env = ClinicalTrialEnv(TASK_MAP["medium"])

def reset():
    global env
    env = ClinicalTrialEnv(TASK_MAP["medium"])
    result = env.reset()
    return {"observation": result["observation"]}

def step(action: int = 0):
    result = env.step(action)
    return {
        "observation": result["observation"],
        "reward": result["reward"],
        "terminated": result["terminated"],
        "truncated": False,
        "info": {}
    }

with gr.Blocks() as demo:
    gr.Markdown("# RL Clinical Trial Optimization Running ✅")

demo.queue().launch(server_name="0.0.0.0", server_port=7860)