import gradio as gr
import matplotlib.pyplot as plt
from inference import run_inference

def normalize_rewards(data):
    rewards = [ep['reward'] for ep in data]
    min_reward = min(rewards) if rewards else 0.0
    max_reward = max(rewards) if rewards else 0.0
    span = max_reward - min_reward + 1e-8
    normalized = []
    for ep in data:
        norm_reward = (ep['reward'] - min_reward) / span
        normalized.append({
            'episode': ep['episode'],
            'reward': norm_reward,
            'assignment_rate': ep['assignment_rate'],
            'diversity_score': ep['diversity_score'],
        })
    return normalized

def run_simulation(task, agent, episodes):
    results = run_inference(task, agent, int(episodes))

    # Text summary
    summary = f"Final Score: {results['score']}\nMean Reward: {results['mean_reward']}\nAssignment Rate: {results['assignment_rate']}\nDiversity Index: {results['diversity_index']}"

    normalized_episodes = normalize_rewards(results['episodes'])

    # Episodes table
    episodes_data = [[ep['episode'], ep['reward'], ep['assignment_rate'], ep['diversity_score']] for ep in normalized_episodes]

    # Graph
    episodes_nums = [ep['episode'] for ep in normalized_episodes]
    rewards = [ep['reward'] for ep in normalized_episodes]
    fig = plt.figure()
    plt.plot(episodes_nums, rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode')
    plt.grid(True)

    return summary, episodes_data, fig

with gr.Blocks(title="AI Clinical Trial Optimization") as demo:
    gr.Markdown("# AI Clinical Trial Optimization")

    with gr.Row():
        task_input = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Task Difficulty")
        agent_input = gr.Dropdown(choices=["random", "rule_based", "greedy_fairness", "q_learning"], value="rule_based", label="Agent Type")
        episodes_input = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Number of Episodes")

    btn = gr.Button("Run Simulation")

    summary_out = gr.Textbox(label="Summary", lines=4)
    table_out = gr.Dataframe(headers=["Episode", "Reward", "Assignment Rate", "Diversity Score"], label="Episode Results")
    plot_out = gr.Plot(label="Reward Progression")

    btn.click(run_simulation, inputs=[task_input, agent_input, episodes_input], outputs=[summary_out, table_out, plot_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
