#!/usr/bin/env python3
"""Automated QA probe for an OpenEnv-compatible API."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class ProbeResult:
    name: str
    ok: bool
    message: str


@dataclass
class EpisodeMetrics:
    rewards: list[float] = field(default_factory=list)
    states: list[list[float]] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    invalid_actions_seen: bool = False
    errors: list[str] = field(default_factory=list)

    def reward_trend(self) -> str:
        if len(self.rewards) < 2:
            return "insufficient-data"
        first = self.rewards[0]
        last = self.rewards[-1]
        if last > first + 1e-9:
            return "increasing"
        if last < first - 1e-9:
            return "decreasing"
        return "flat"

    def state_changes(self) -> bool:
        if len(self.states) < 2:
            return False
        return any(self.states[i] != self.states[i - 1] for i in range(1, len(self.states)))

    def reward_all_zero(self) -> bool:
        return bool(self.rewards) and all(abs(value) < 1e-9 for value in self.rewards)

    def done_reached(self) -> bool:
        return any(self.dones)


class OpenEnvAPITester:
    def __init__(self, base_url: str, timeout: float, max_steps: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_steps = max_steps
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _safe_json(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except Exception:
            return None

    def _post(self, path: str, payload: Any = None) -> tuple[bool, int | None, Any, str | None]:
        try:
            if payload is None:
                response = self.session.post(self._url(path), timeout=self.timeout)
            else:
                response = self.session.post(self._url(path), json=payload, timeout=self.timeout)
            return True, response.status_code, self._safe_json(response), None
        except requests.RequestException as exc:
            return False, None, None, str(exc)

    def check_reset_without_body(self) -> ProbeResult:
        ok, status, data, error = self._post("/reset")
        if not ok:
            return ProbeResult("reset_without_body", False, f"Request failed: {error}")
        if status != 200:
            return ProbeResult("reset_without_body", False, f"Expected 200, got {status}. Body: {data}")
        valid, message = self._validate_reset_payload(data)
        return ProbeResult("reset_without_body", valid, message)

    def check_reset_with_body(self) -> ProbeResult:
        ok, status, data, error = self._post("/reset", {"seed": 42, "task": "medium"})
        if not ok:
            return ProbeResult("reset_with_body", False, f"Request failed: {error}")
        if status != 200:
            return ProbeResult("reset_with_body", False, f"Expected 200, got {status}. Body: {data}")
        valid, message = self._validate_reset_payload(data)
        return ProbeResult("reset_with_body", valid, message)

    def _validate_reset_payload(self, data: Any) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return False, f"Reset response must be a JSON object, got {type(data).__name__}"
        if "observation" not in data:
            return False, "Reset response missing observation"
        if "state" not in data:
            return False, "Reset response missing state"
        state = data.get("state")
        if not isinstance(state, list):
            return False, "Reset state must be a list"
        if len(state) == 0:
            return False, "Reset state is empty"
        if all(value == 0 for value in state):
            return False, "Reset state is all zeros"
        return True, "Reset response structure is valid"

    def _extract_trial_ids(self, reset_payload: dict[str, Any]) -> list[int]:
        observation = reset_payload.get("observation", {}) or {}
        trials = observation.get("trials", []) or []
        trial_ids = []
        for trial in trials:
            try:
                trial_id = int(trial.get("trial_id"))
                if trial_id > 0:
                    trial_ids.append(trial_id)
            except Exception:
                continue
        return trial_ids

    def run_episode(self) -> EpisodeMetrics:
        metrics = EpisodeMetrics()

        ok, status, reset_data, error = self._post("/reset", {"seed": 7, "task": "medium"})
        if not ok:
            metrics.errors.append(f"Reset failed: {error}")
            return metrics
        if status != 200 or not isinstance(reset_data, dict):
            metrics.errors.append(f"Reset returned status {status} with body {reset_data}")
            return metrics

        trial_ids = self._extract_trial_ids(reset_data)
        if not trial_ids:
            trial_ids = [1]

        current_state = reset_data.get("state", [])
        seen_states = {tuple(current_state) if isinstance(current_state, list) else ()}
        step_count = 0
        done = bool(reset_data.get("done", False))

        while not done and step_count < self.max_steps:
            action = trial_ids[step_count % len(trial_ids)]
            ok, step_status, step_data, error = self._post("/step", {"action": action})
            if not ok:
                metrics.errors.append(f"Step request failed on action {action}: {error}")
                break
            if step_status != 200 or not isinstance(step_data, dict):
                metrics.errors.append(f"Step returned status {step_status} with body {step_data}")
                break

            reward = float(step_data.get("reward", 0.0))
            done = bool(step_data.get("done", False))
            state = step_data.get("state", [])
            if isinstance(state, list):
                metrics.states.append(state)
                seen_states.add(tuple(state))
            metrics.rewards.append(reward)
            metrics.dones.append(done)
            step_count += 1

        metrics.invalid_actions_seen = self._probe_invalid_actions()
        if len(seen_states) > 1:
            pass
        else:
            metrics.errors.append("State did not change across episode steps")

        return metrics

    def _probe_invalid_actions(self) -> bool:
        ok, status, data, error = self._post("/reset", {"seed": 99, "task": "medium"})
        if not ok or status != 200 or not isinstance(data, dict):
            return False

        ok, invalid_status, invalid_data, invalid_error = self._post("/step", {"action": -1})
        if not ok:
            return False
        if invalid_status == 422:
            return True
        if invalid_status == 200 and isinstance(invalid_data, dict):
            info = invalid_data.get("info", {}) or {}
            reason = str(info.get("reason", "")).lower()
            return "invalid" in reason or "error" in reason
        return False


def print_result(result: ProbeResult) -> None:
    prefix = "✅" if result.ok else "❌"
    print(f"{prefix} {result.name}: {result.message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="QA probe for an OpenEnv-compatible API.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps to run in one episode")
    args = parser.parse_args()

    tester = OpenEnvAPITester(args.base_url, args.timeout, args.max_steps)
    results: list[ProbeResult] = []

    print(f"OpenEnv QA Probe -> {args.base_url}")
    print("-" * 60)

    reset_without_body = tester.check_reset_without_body()
    reset_with_body = tester.check_reset_with_body()
    results.extend([reset_without_body, reset_with_body])
    print_result(reset_without_body)
    print_result(reset_with_body)

    episode = tester.run_episode()

    step_ok = len(episode.rewards) > 0 and not episode.errors
    step_message = f"{len(episode.rewards)} steps, last reward={episode.rewards[-1]:.4f}" if episode.rewards else "No successful steps"
    results.append(ProbeResult("step_loop", step_ok, step_message))
    print_result(results[-1])

    state_changes = episode.state_changes()
    results.append(ProbeResult("state_changes", state_changes, "State changes detected" if state_changes else "State stayed constant"))
    print_result(results[-1])

    reward_non_zero = not episode.reward_all_zero()
    results.append(ProbeResult("reward_values", reward_non_zero, "Rewards are non-zero" if reward_non_zero else "Reward always zero"))
    print_result(results[-1])

    done_reached = episode.done_reached()
    results.append(ProbeResult("done_flag", done_reached, "Done reached" if done_reached else "Done never reached"))
    print_result(results[-1])

    invalid_action_handled = episode.invalid_actions_seen
    results.append(ProbeResult("invalid_actions", invalid_action_handled, "Invalid action handling detected" if invalid_action_handled else "Invalid actions not confirmed"))
    print_result(results[-1])

    if episode.rewards:
        trend = episode.reward_trend()
        print(f"📈 Reward trend: {trend}")
    if episode.errors:
        print("\nIssues detected:")
        for item in episode.errors:
            print(f"- {item}")

    pass_count = sum(1 for item in results if item.ok)
    fail_count = len(results) - pass_count
    environment_status = "PASS" if fail_count == 0 else "FAIL"

    print("\n" + "=" * 60)
    print(f"ENVIRONMENT STATUS: {environment_status}")
    print(f"Checks passed: {pass_count}/{len(results)}")
    if fail_count:
        print("Suggested fixes:")
        if not reset_without_body.ok:
            print("- Make /reset body optional and parse Request manually.")
        if not reset_with_body.ok:
            print("- Validate JSON body parsing and default task/seed handling.")
        if not state_changes:
            print("- Ensure state changes on every valid step.")
        if not reward_non_zero:
            print("- Adjust reward shaping so valid actions produce non-zero rewards.")
        if not done_reached:
            print("- Ensure the episode termination condition is reachable.")
        if not invalid_action_handled:
            print("- Confirm invalid actions return 4xx or an explicit error reason.")
    print("=" * 60)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
