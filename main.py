"""
CLI runner — no UI required.
Usage:
  python main.py [task] [agent] [episodes]
  python main.py medium rule_based 10
  python main.py hard greedy_fairness 5
  python main.py easy q_learning 100
"""
import sys
import matplotlib.pyplot as plt
from env    import ClinicalTrialEnv
from agent  import RandomAgent, RuleBasedAgent, GreedyFairnessAgent, QLearningAgent
from tasks  import TASK_MAP
from grader import grade


def build_agent(name: str, n_trials: int):
    name = name.lower().replace("-", "_")
    if name == "random":
        return RandomAgent(n_trials)
    if name == "greedy_fairness":
        return GreedyFairnessAgent()
    if name == "q_learning":
        return QLearningAgent(n_trials)
    return RuleBasedAgent()


def run(task="medium", agent_name="rule_based", episodes=200):
    config = dict(TASK_MAP.get(task, TASK_MAP["medium"]))
    env    = ClinicalTrialEnv(config)
    agent  = build_agent(agent_name, config["n_trials"])

    W = 70
    print("=" * W)
    print(f"  Training RL Agent — {task.upper()} | {agent.name} | {episodes} episodes")
    print("=" * W)

    episode_rewards = []
    assignment_rates = []
    diversity_scores = []

    for ep in range(episodes):
        result = env.reset(seed=ep)
        obs    = result["observation"]
        total_reward = 0.0

        while True:
            action = agent.act(obs)
            result = env.step(action)
            next_obs = result["observation"]
            reward = result["reward"]
            total_reward += reward

            # Learn if agent has learn method
            if hasattr(agent, 'learn'):
                agent.learn(obs, action, reward, next_obs, result["terminated"])

            obs = next_obs

            if result["terminated"]:
                break

        sys_s = obs["system"]
        assignment_rate = sys_s['patients_assigned'] / max(sys_s['total_patients'], 1)
        diversity_score = sys_s['diversity_index']
        episode_rewards.append(total_reward)
        assignment_rates.append(assignment_rate)
        diversity_scores.append(diversity_score)

        # Update epsilon if agent has it
        if hasattr(agent, 'update_epsilon'):
            agent.update_epsilon()

        # Per-episode logging
        print(f"Episode {ep+1:3d}: Reward={total_reward:+.2f}, Assignment Rate={assignment_rate:.3f}, Diversity={diversity_score:.3f}")

    # Moving average for smoothing
    def moving_average(data, window=10):
        return [sum(data[max(0, i-window+1):i+1]) / len(data[max(0, i-window+1):i+1]) for i in range(len(data))]

    smoothed_rewards = moving_average(episode_rewards)
    smoothed_assignments = moving_average(assignment_rates)
    smoothed_diversities = moving_average(diversity_scores)

    # Plot rewards
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, episodes+1), episode_rewards, alpha=0.3, label='Raw')
    plt.plot(range(1, episodes+1), smoothed_rewards, label='Smoothed')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(1, episodes+1), assignment_rates, alpha=0.3, label='Raw')
    plt.plot(range(1, episodes+1), smoothed_assignments, label='Smoothed')
    plt.xlabel('Episode')
    plt.ylabel('Assignment Rate')
    plt.title('Assignment Rates')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(1, episodes+1), diversity_scores, alpha=0.3, label='Raw')
    plt.plot(range(1, episodes+1), smoothed_diversities, label='Smoothed')
    plt.xlabel('Episode')
    plt.ylabel('Diversity Index')
    plt.title('Diversity Scores')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300)
    # plt.show()  # Disabled to prevent blocking API response

    # Final performance summary
    print("-" * W)
    print("FINAL PERFORMANCE SUMMARY:")
    print(f"  Average Reward: {sum(episode_rewards)/episodes:+.2f}")
    print(f"  Average Assignment Rate: {sum(assignment_rates)/episodes:.3f}")
    print(f"  Average Diversity: {sum(diversity_scores)/episodes:.3f}")
    print(f"  Best Reward: {max(episode_rewards):+.2f}")
    print(f"  Best Assignment Rate: {max(assignment_rates):.3f}")
    print(f"  Best Diversity: {max(diversity_scores):.3f}")
    print("-" * W)

    # Evaluate final performance
    scores = grade(env, agent, episodes=10)
    print(f"\nEVALUATION (10 test episodes):")
    print(f"    Overall score   : {scores['score']:.4f} / 1.0000")
    print(f"    Assignment rate : {scores['assignment_rate']:.4f}")
    print(f"    Diversity index : {scores['diversity_index']:.4f}")
    print(f"    Fill rate       : {scores['fill_rate']:.4f}")
    print(f"    Mean reward     : {scores['mean_reward']:.4f}")
    print("=" * W)
    return scores


if __name__ == "__main__":
    task_arg     = sys.argv[1] if len(sys.argv) > 1 else "medium"
    agent_arg    = sys.argv[2] if len(sys.argv) > 2 else "rule_based"
    episodes_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    run(task_arg, agent_arg, episodes_arg)
