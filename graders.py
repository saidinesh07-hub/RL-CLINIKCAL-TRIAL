"""
Task-specific graders for OpenEnv compliance.

Each grader evaluates simulation output and returns a score in [0.0, 1.0].
"""
from env import ClinicalTrialEnv


def grade_reward(env: ClinicalTrialEnv, agent, episodes: int = 50) -> dict:
    """
    Task 1: Reward Optimization
    Maximize cumulative reward across episodes.
    Score = normalized mean episode reward (0.0 → 1.0)
    """
    episode_rewards = []

    for ep in range(episodes):
        result = env.reset(seed=ep)
        obs = result["observation"]
        ep_reward = 0.0

        while True:
            action = agent.act(obs)
            result = env.step(action)
            obs = result["observation"]
            ep_reward += result["reward"]

            if hasattr(agent, "learn"):
                agent.learn(obs, action, result["reward"], obs, result["terminated"])

            if result["terminated"]:
                break

        if hasattr(agent, "update_epsilon"):
            agent.update_epsilon()

        episode_rewards.append(ep_reward)

    mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0

    # Normalize reward to [0, 1]: assume typical range is -5 to +20
    normalized_score = max(0.0, min(1.0, (mean_reward + 5.0) / 25.0))

    return {
        "score": round(normalized_score, 4),
        "mean_reward": round(mean_reward, 4),
        "max_reward": round(max(episode_rewards), 4) if episode_rewards else 0.0,
        "min_reward": round(min(episode_rewards), 4) if episode_rewards else 0.0,
    }


def grade_diversity(env: ClinicalTrialEnv, agent, episodes: int = 50) -> dict:
    """
    Task 2: Fairness Optimization
    Maintain high diversity score (target > 0.8).
    Score = average diversity index across episodes
    """
    diversity_scores = []

    for ep in range(episodes):
        result = env.reset(seed=ep)
        obs = result["observation"]

        while True:
            action = agent.act(obs)
            result = env.step(action)
            obs = result["observation"]

            if hasattr(agent, "learn"):
                agent.learn(obs, action, result["reward"], obs, result["terminated"])

            if result["terminated"]:
                break

        if hasattr(agent, "update_epsilon"):
            agent.update_epsilon()

        diversity_index = obs["system"]["diversity_index"]
        diversity_scores.append(diversity_index)

    mean_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0

    # Score based on how well diversity meets the > 0.8 target
    # If diversity >= 0.8, score = 1.0
    # If diversity < 0.8, score = diversity (linear)
    if mean_diversity >= 0.8:
        score = 1.0
    else:
        score = mean_diversity

    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "mean_diversity": round(mean_diversity, 4),
        "max_diversity": round(max(diversity_scores), 4) if diversity_scores else 0.0,
        "min_diversity": round(min(diversity_scores), 4) if diversity_scores else 0.0,
        "target_met": mean_diversity >= 0.8,
    }


def grade_assignment(env: ClinicalTrialEnv, agent, episodes: int = 50) -> dict:
    """
    Task 3: Balanced Assignment
    Maintain stable assignment rate and resource utilization.
    Score = stability of assignment rate + average fill rate
    """
    assignment_rates = []
    fill_rates = []

    for ep in range(episodes):
        result = env.reset(seed=ep)
        obs = result["observation"]

        while True:
            action = agent.act(obs)
            result = env.step(action)
            obs = result["observation"]

            if hasattr(agent, "learn"):
                agent.learn(obs, action, result["reward"], obs, result["terminated"])

            if result["terminated"]:
                break

        if hasattr(agent, "update_epsilon"):
            agent.update_epsilon()

        sys_state = obs["system"]
        n_total = max(sys_state["total_patients"], 1)
        assignment_rate = sys_state["patients_assigned"] / n_total
        assignment_rates.append(assignment_rate)

        # Compute average fill rate across trials
        avg_fill = sum(t["fill_rate"] for t in obs["trials"]) / max(len(obs["trials"]), 1)
        fill_rates.append(avg_fill)

    mean_assignment_rate = sum(assignment_rates) / len(assignment_rates) if assignment_rates else 0.0
    mean_fill_rate = sum(fill_rates) / len(fill_rates) if fill_rates else 0.0

    # Calculate stability (low variance = high stability)
    if len(assignment_rates) > 1:
        variance = sum((x - mean_assignment_rate) ** 2 for x in assignment_rates) / len(assignment_rates)
        std_dev = variance ** 0.5
        stability = max(0.0, 1.0 - std_dev)  # High std dev = low stability
    else:
        stability = 1.0

    # Score = 0.6 * assignment_rate + 0.2 * stability + 0.2 * fill_rate
    score = max(0.0, min(1.0, 0.6 * mean_assignment_rate + 0.2 * stability + 0.2 * mean_fill_rate))

    return {
        "score": round(score, 4),
        "mean_assignment_rate": round(mean_assignment_rate, 4),
        "stability": round(stability, 4),
        "mean_fill_rate": round(mean_fill_rate, 4),
        "std_dev_assignment": round(std_dev if len(assignment_rates) > 1 else 0.0, 4),
    }


def grade(env: ClinicalTrialEnv, agent, episodes: int = 50) -> dict:
    """
    Legacy unified grader - kept for backward compatibility.
    Returns overall score combining all three task objectives.
    """
    reward_result = grade_reward(env, agent, episodes)
    diversity_result = grade_diversity(env, agent, episodes)
    assignment_result = grade_assignment(env, agent, episodes)

    # Weighted combination
    combined_score = (
        0.35 * reward_result["score"]
        + 0.35 * diversity_result["score"]
        + 0.30 * assignment_result["score"]
    )

    return {
        "score": round(max(0.0, min(1.0, combined_score)), 4),
        "reward_score": reward_result["score"],
        "diversity_score": diversity_result["score"],
        "assignment_score": assignment_result["score"],
        "assignment_rate": assignment_result["mean_assignment_rate"],
        "diversity_index": diversity_result["mean_diversity"],
        "fill_rate": assignment_result["mean_fill_rate"],
        "mean_reward": reward_result["mean_reward"],
    }
