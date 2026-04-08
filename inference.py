#!/usr/bin/env python3
"""
OpenEnv Inference Script
Runs simulation without UI, outputs structured logs via stdout
Usage:
  python inference.py [task] [agent] [episodes]
  python inference.py medium rule_based 10
"""
import sys
import os
from env import ClinicalTrialEnv
from agent import RandomAgent, RuleBasedAgent, GreedyFairnessAgent, QLearningAgent
from tasks import TASK_MAP

# Environment variable support
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "clinical-trial-rl")
HF_TOKEN = os.getenv("HF_TOKEN", "")


def build_agent(name: str, n_trials: int):
    """Instantiate agent from name."""
    name = name.lower().replace("-", "_")
    if name == "random":
        return RandomAgent(n_trials)
    if name == "greedy_fairness":
        return GreedyFairnessAgent()
    if name == "q_learning":
        return QLearningAgent(n_trials)
    return RuleBasedAgent()


def run_inference(task: str = "medium", agent_name: str = "rule_based", episodes: int = 10) -> dict:
    """
    Run simulation and return structured results.
    Returns dict with overall score and per-episode metrics.
    """
    config = dict(TASK_MAP.get(task, TASK_MAP["medium"]))
    env = ClinicalTrialEnv(config)
    agent = build_agent(agent_name, config["n_trials"])

    episode_metrics = []
    total_rewards = []
    total_assignments = []
    total_diversities = []
    episode_data = []

    for ep in range(episodes):
        result = env.reset(seed=ep)
        obs = result["observation"]
        ep_reward = 0.0
        step_count = 0

        while True:
            step_count += 1
            action = agent.act(obs)
            result = env.step(action)
            next_obs = result["observation"]
            reward = result["reward"]
            ep_reward += reward

            print(f"[STEP] episode={ep+1} step={step_count} reward={reward:.4f}")

            if hasattr(agent, "learn"):
                agent.learn(obs, action, reward, next_obs, result["terminated"])

            obs = next_obs

            if result["terminated"]:
                break

        if hasattr(agent, "update_epsilon"):
            agent.update_epsilon()

        sys_state = obs["system"]
        assignment_rate = sys_state["patients_assigned"] / max(sys_state["total_patients"], 1)
        diversity_score = sys_state["diversity_index"]

        total_rewards.append(ep_reward)
        total_assignments.append(assignment_rate)
        total_diversities.append(diversity_score)

        episode_data.append({
            "episode": ep + 1,
            "reward": ep_reward,
            "assignment_rate": assignment_rate,
            "diversity_score": diversity_score,
        })

    # Output START marker
    print("[START]")

    # Episodes are run above with prints

    # Populate episode_metrics
    for data in episode_data:
        episode_metrics.append({
            "episode": data["episode"],
            "reward": round(data["reward"], 4),
            "assignment_rate": round(data["assignment_rate"], 4),
            "diversity_score": round(data["diversity_score"], 4),
        })

    # Compute final score (weighted average)
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    avg_assignment = sum(total_assignments) / len(total_assignments) if total_assignments else 0.0
    avg_diversity = sum(total_diversities) / len(total_diversities) if total_diversities else 0.0

    # Final score formula: 0.5 * assignment + 0.3 * diversity + 0.2 * normalized_reward
    # Normalize reward to [0, 1] range (assuming typical range is -5 to +20)
    normalized_reward = max(0.0, min(1.0, (avg_reward + 5.0) / 25.0))
    final_score = max(0.0, min(1.0, 0.5 * avg_assignment + 0.3 * avg_diversity + 0.2 * normalized_reward))

    # Output END marker with final score
    print(f"[END] score={final_score:.4f}")

    return {
        "score": round(final_score, 4),
        "assignment_rate": round(avg_assignment, 4),
        "diversity_index": round(avg_diversity, 4),
        "mean_reward": round(avg_reward, 4),
        "normalized_reward": round(normalized_reward, 4),
        "episodes": episode_metrics,
    }


if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else "medium"
    agent_arg = sys.argv[2] if len(sys.argv) > 2 else "rule_based"
    episodes_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    results = run_inference(task_arg, agent_arg, episodes_arg)
    sys.exit(0)
