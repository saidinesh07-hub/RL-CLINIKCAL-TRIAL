from env import ClinicalTrialEnv


def grade(env: ClinicalTrialEnv, agent, episodes: int = 50) -> dict:
    """
    Returns a dict with overall score [0,1] and sub-scores.
    score = 0.5 * assignment_rate + 0.3 * diversity_index + 0.2 * fill_rate
    """
    assignment_rates = []
    diversity_scores = []
    fill_scores      = []
    total_rewards    = []

    for ep in range(episodes):
        result = env.reset(seed=ep)
        obs    = result["observation"]
        ep_reward = 0.0

        while True:
            action = agent.act(obs)
            result = env.step(action)
            obs       = result["observation"]
            ep_reward += result["reward"]
            if result["terminated"] or result["truncated"]:
                break

        sys_state = obs["system"]
        n_total   = max(sys_state["total_patients"], 1)

        assignment_rates.append(sys_state["patients_assigned"] / n_total)
        diversity_scores.append(sys_state["diversity_index"])
        fill_scores.append(
            sum(t["fill_rate"] for t in obs["trials"]) / max(len(obs["trials"]), 1)
        )
        total_rewards.append(ep_reward)

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    ar = mean(assignment_rates)
    di = mean(diversity_scores)
    fs = mean(fill_scores)
    score = max(0.0, min(1.0, 0.5 * ar + 0.3 * di + 0.2 * fs))

    return {
        "score":            round(score, 4),
        "assignment_rate":  round(ar, 4),
        "diversity_index":  round(di, 4),
        "fill_rate":        round(fs, 4),
        "mean_reward":      round(mean(total_rewards), 4),
    }
