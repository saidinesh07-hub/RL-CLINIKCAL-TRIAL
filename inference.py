#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from typing import Any
from urllib import request

from agent import GreedyFairnessAgent, QLearningAgent, RandomAgent, RuleBasedAgent
from env import ClinicalTrialEnv
from tasks import TASK_MAP


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def build_agent(name: str, n_trials: int):
    normalized = name.lower().replace("-", "_")
    if normalized == "random":
        return RandomAgent(n_trials)
    if normalized == "greedy_fairness":
        return GreedyFairnessAgent()
    if normalized == "q_learning":
        return QLearningAgent(n_trials)
    return RuleBasedAgent()


def _api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{API_BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def run_local(task: str, agent_name: str, episodes: int) -> dict[str, Any]:
    config = dict(TASK_MAP.get(task, TASK_MAP["medium"]))
    env = ClinicalTrialEnv(config)
    agent = build_agent(agent_name, config["n_trials"])

    traces: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    episodes_output: list[dict[str, Any]] = []

    for ep in range(max(1, episodes)):
        reset_result = env.reset(seed=ep)
        obs = reset_result["observation"]
        done = False
        ep_reward = 0.0
        ep_actions: list[int] = []

        while not done:
            action = int(agent.act(obs))
            step_result = env.step(action)
            next_obs = step_result["observation"]
            reward = float(step_result["reward"])
            done = bool(step_result["terminated"] or step_result.get("truncated", False))

            traces.append(
                {
                    "episode": ep + 1,
                    "step": int(next_obs.get("system", {}).get("step", 0)),
                    "action": action,
                    "reward": reward,
                    "done": done,
                }
            )

            ep_actions.append(action)
            ep_reward += reward

            if hasattr(agent, "learn"):
                agent.learn(obs, action, reward, next_obs, bool(step_result["terminated"]))
            obs = next_obs

        if hasattr(agent, "update_epsilon"):
            agent.update_epsilon()

        system = obs.get("system", {})
        episode_summaries.append(
            {
                "episode": ep + 1,
                "total_reward": round(ep_reward, 4),
                "steps": int(system.get("step", 0)),
                "patients_assigned": int(system.get("patients_assigned", 0)),
                "patients_rejected": int(system.get("patients_rejected", 0)),
                "actions": ep_actions,
            }
        )
        steps = max(int(system.get("step", 0)), 1)
        episodes_output.append(
            {
                "episode": ep + 1,
                "reward": round(ep_reward, 4),
                "assignment_rate": round(int(system.get("patients_assigned", 0)) / max(int(system.get("total_patients", 1)), 1), 4),
                "diversity_score": round(float(system.get("diversity_index", 0.0)), 4),
                "avg_step_reward": round(ep_reward / steps, 4),
            }
        )

    rewards = [item["reward"] for item in traces]
    non_zero_rewards = sum(1 for r in rewards if abs(r) > 1e-9)
    avg_assignment = (
        sum(item["assignment_rate"] for item in episodes_output) / max(len(episodes_output), 1)
    )
    avg_diversity = (
        sum(item["diversity_score"] for item in episodes_output) / max(len(episodes_output), 1)
    )
    avg_reward = sum(rewards) / max(len(rewards), 1)
    normalized_reward = max(0.0, min(1.0, (avg_reward + 5.0) / 25.0))
    score = max(0.0, min(1.0, 0.5 * avg_assignment + 0.3 * avg_diversity + 0.2 * normalized_reward))

    return {
        "mode": "local",
        "episode_count": len(episode_summaries),
        "steps": len(traces),
        "non_zero_rewards": non_zero_rewards,
        "score": round(score, 4),
        "assignment_rate": round(avg_assignment, 4),
        "diversity_index": round(avg_diversity, 4),
        "mean_reward": round(avg_reward, 4),
        "normalized_reward": round(normalized_reward, 4),
        "episodes": episodes_output,
        "traces": traces,
        "episode_summaries": episode_summaries,
    }


def run_via_api(task: str, episodes: int) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    episodes_output: list[dict[str, Any]] = []

    for ep in range(max(1, episodes)):
        reset_response = _api_post("/reset", {"seed": ep, "task": task})
        obs = reset_response.get("observation", {})
        done = bool(reset_response.get("done", False))
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            rec_action = int((obs.get("system", {}) or {}).get("recommended_action", 0))
            step_response = _api_post("/step", {"action": rec_action})
            obs = step_response.get("observation", {})
            reward = float(step_response.get("reward", 0.0))
            done = bool(step_response.get("done", False))
            ep_reward += reward
            ep_steps += 1

            traces.append(
                {
                    "episode": ep + 1,
                    "step": int((obs.get("system", {}) or {}).get("step", 0)),
                    "action": rec_action,
                    "reward": reward,
                    "done": done,
                }
            )

        system = (obs.get("system", {}) or {})
        episodes_output.append(
            {
                "episode": ep + 1,
                "reward": round(ep_reward, 4),
                "assignment_rate": round(int(system.get("patients_assigned", 0)) / max(int(system.get("total_patients", 1)), 1), 4),
                "diversity_score": round(float(system.get("diversity_index", 0.0)), 4),
                "avg_step_reward": round(ep_reward / max(ep_steps, 1), 4),
            }
        )

    rewards = [item["reward"] for item in traces]
    non_zero_rewards = sum(1 for r in rewards if abs(r) > 1e-9)
    avg_assignment = (
        sum(item["assignment_rate"] for item in episodes_output) / max(len(episodes_output), 1)
    )
    avg_diversity = (
        sum(item["diversity_score"] for item in episodes_output) / max(len(episodes_output), 1)
    )
    avg_reward = sum(rewards) / max(len(rewards), 1)
    normalized_reward = max(0.0, min(1.0, (avg_reward + 5.0) / 25.0))
    score = max(0.0, min(1.0, 0.5 * avg_assignment + 0.3 * avg_diversity + 0.2 * normalized_reward))

    return {
        "mode": "api",
        "episode_count": max(1, episodes),
        "steps": len(traces),
        "non_zero_rewards": non_zero_rewards,
        "score": round(score, 4),
        "assignment_rate": round(avg_assignment, 4),
        "diversity_index": round(avg_diversity, 4),
        "mean_reward": round(avg_reward, 4),
        "normalized_reward": round(normalized_reward, 4),
        "episodes": episodes_output,
        "traces": traces,
    }


def run_inference(
    task: str = "medium",
    agent_name: str = "rule_based",
    episodes: int = 50,
    use_api: bool = False,
) -> dict[str, Any]:
    if use_api:
        return run_via_api(task=task, episodes=episodes)
    return run_local(task=task, agent_name=agent_name, episodes=episodes)


if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else "medium"
    agent_arg = sys.argv[2] if len(sys.argv) > 2 else "rule_based"
    episodes_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    mode_arg = (sys.argv[4] if len(sys.argv) > 4 else "local").lower()
    use_api_arg = mode_arg == "api"

    result = run_inference(task=task_arg, agent_name=agent_arg, episodes=episodes_arg, use_api=use_api_arg)
    print(json.dumps(result, indent=2))
