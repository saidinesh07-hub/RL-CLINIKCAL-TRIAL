from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from tasks import TASK_MAP

CONDITIONS = ["oncology", "cardiology", "neurology", "immunology"]
AGE_GROUPS = ["pediatric", "adult", "elderly"]
GENDERS = ["M", "F", "other"]
COMORBIDS = ["diabetes", "hypertension", "renal_failure", "liver_disease"]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


@dataclass
class Patient:
    patient_id: str
    condition: str
    severity: int
    age_group: str
    gender: str
    comorbidities: list
    wait_steps: int = 0
    prior_rejection: bool = False


@dataclass
class Trial:
    trial_id: int
    condition: str
    required_severity: list
    excluded_comorbidities: list
    capacity_total: int
    capacity_used: int = 0
    priority: int = 1
    diversity_target: dict = field(default_factory=dict)
    enrolled_age_groups: dict = field(
        default_factory=lambda: {"pediatric": 0, "adult": 0, "elderly": 0}
    )

    @property
    def is_full(self) -> bool:
        return self.capacity_used >= self.capacity_total

    @property
    def fill_rate(self) -> float:
        return self.capacity_used / self.capacity_total if self.capacity_total > 0 else 0.0

    @property
    def slots_remaining(self) -> int:
        return max(0, self.capacity_total - self.capacity_used)


class ClinicalTrialEnv:
    metadata = {"version": "1.0.0"}

    def __init__(self, config: dict | str):
        config = self._resolve_config(config)

        self.n_patients = config.get("n_patients", 50)
        self.n_trials = config.get("n_trials", 6)
        self.capacity_range = config.get("capacity_range", (8, 15))
        self.difficulty = config.get("difficulty", "medium")
        self.fairness_weight = config.get("fairness_weight", 0.1)
        self.seed = config.get("seed", 0)

        self._rng = random.Random(self.seed)
        self.trials: list[Trial] = []
        self.patient_queue: list[Patient] = []
        self._step = 0
        self._assigned = 0
        self._rejected = 0
        self._invalid_actions = 0
        self._done = False
        self._episode_reward_total = 0.0
        self._history: list[dict] = []

    @staticmethod
    def _resolve_config(config: dict | str) -> dict:
        if isinstance(config, str):
            if config not in TASK_MAP:
                raise ValueError(
                    f"Unknown difficulty '{config}'. Expected one of: {', '.join(TASK_MAP.keys())}"
                )
            return dict(TASK_MAP[config])
        if not isinstance(config, dict):
            raise TypeError("config must be a dict or one of: easy, medium, hard")
        return dict(config)

    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)

        self.trials = self._generate_trials()
        self.patient_queue = self._generate_patients()
        self._step = 0
        self._assigned = 0
        self._rejected = 0
        self._invalid_actions = 0
        self._done = False
        self._episode_reward_total = 0.0
        self._history = []

        return {"observation": self.state()}

    def step(self, action: int) -> dict:
        if self._done or self._step >= len(self.patient_queue):
            self._done = True
            return {
                "observation": self.state(),
                "reward": 0.0,
                "terminated": True,
                "truncated": False,
                "info": {"reason": "episode_finished"},
            }

        patient = self.patient_queue[self._step]
        action_value, action_valid = self._coerce_action(action)

        if not action_valid:
            reward, info = self._handle_invalid_action(patient, action)
        else:
            reward, info = self._compute_reward_and_apply(patient, action_value)

        self._advance_wait_times()
        self._step += 1

        terminated = self._step >= len(self.patient_queue) or all(trial.is_full for trial in self.trials)
        if terminated:
            self._done = True
            bonus = self._end_of_episode_bonus()
            reward = _clamp(reward + bonus)
            info["episode_bonus"] = round(bonus, 4)

        reward = _clamp(reward)
        self._episode_reward_total += reward

        self._history.append(
            {
                "step": self._step - 1,
                "patient_id": patient.patient_id,
                "condition": patient.condition,
                "severity": patient.severity,
                "action": action_value if action_valid else action,
                "reward": round(reward, 4),
                "events": info.get("events", []),
                "reason": info.get("reason", ""),
            }
        )

        return {
            "observation": self.state(),
            "reward": float(round(reward, 4)),
            "terminated": terminated,
            "truncated": False,
            "info": info,
        }

    def state(self) -> dict:
        if self._step >= len(self.patient_queue):
            current_patient = {}
            recommendation = {
                "recommended_action": 0,
                "recommended_reason": "Episode complete. Reset to continue.",
                "valid_trials": [],
                "trial_statuses": [],
            }
        else:
            patient = self.patient_queue[self._step]
            recommended_action, recommended_reason, trial_statuses = self._recommendation(patient)
            current_patient = {
                "patient_id": patient.patient_id,
                "condition": patient.condition,
                "severity": patient.severity,
                "age_group": patient.age_group,
                "gender": patient.gender,
                "comorbidities": list(patient.comorbidities),
                "wait_steps": patient.wait_steps,
                "prior_rejection": patient.prior_rejection,
            }
            recommendation = {
                "recommended_action": recommended_action,
                "recommended_reason": recommended_reason,
                "valid_trials": [item["trial_id"] for item in trial_statuses if item["valid"]],
                "trial_statuses": trial_statuses,
            }

        accepted = self._assigned
        processed = max(self._step, 1)
        total_patients = max(len(self.patient_queue), 1)
        avg_fill = sum(trial.fill_rate for trial in self.trials) / max(len(self.trials), 1)
        diversity_index = self._compute_diversity_index()
        acceptance_rate = accepted / processed

        return {
            "patient": current_patient,
            "trials": [
                {
                    "trial_id": trial.trial_id,
                    "condition": trial.condition,
                    "required_severity": list(trial.required_severity),
                    "excluded_comorbidities": list(trial.excluded_comorbidities),
                    "capacity_total": trial.capacity_total,
                    "capacity_used": trial.capacity_used,
                    "slots_remaining": trial.slots_remaining,
                    "is_full": trial.is_full,
                    "fill_rate": float(round(trial.fill_rate, 3)),
                    "priority": trial.priority,
                    "diversity_target": dict(trial.diversity_target),
                }
                for trial in self.trials
            ],
            "system": {
                "step": self._step,
                "total_patients": len(self.patient_queue),
                "patients_assigned": self._assigned,
                "patients_rejected": self._rejected,
                "invalid_actions": self._invalid_actions,
                "acceptance_rate": float(round(acceptance_rate, 3)),
                "assignment_rate": float(round(self._assigned / total_patients, 3)),
                "average_reward": float(round(self._episode_reward_total / processed, 3)),
                "average_fill_rate": float(round(avg_fill, 3)),
                "diversity_index": float(round(diversity_index, 3)),
                "done": self._done,
                **recommendation,
            },
        }

    def _coerce_action(self, action: int) -> tuple[int, bool]:
        try:
            action_value = int(action)
        except (TypeError, ValueError):
            return 0, False
        if not 0 <= action_value <= self.n_trials:
            return action_value, False
        return action_value, True

    def _handle_invalid_action(self, patient: Patient, action) -> tuple[float, dict]:
        self._invalid_actions += 1
        return 0.0, {
            "action": action,
            "patient_id": patient.patient_id,
            "events": ["invalid_action"],
            "reason": "invalid_action",
        }

    def _compute_reward_and_apply(self, patient: Patient, action: int) -> tuple[float, dict]:
        info = {"action": action, "patient_id": patient.patient_id, "events": []}

        if action == 0:
            eligible = self._eligible_trials(patient)
            patient.prior_rejection = True
            self._rejected += 1
            if eligible:
                reward = 0.18
                info["events"].append("unjustified_rejection")
                info["reason"] = "unjustified_rejection"
            else:
                reward = 0.72
                info["events"].append("valid_rejection")
                info["reason"] = "valid_rejection"
            reward -= min(0.01 * patient.wait_steps, 0.08)
            return _clamp(reward), info

        trial = next((item for item in self.trials if item.trial_id == action), None)
        if trial is None:
            self._invalid_actions += 1
            return 0.0, {"events": ["invalid_trial"], "reason": "invalid_trial"}

        valid, reasons, score = self._assess_trial(patient, trial)
        if not valid:
            self._invalid_actions += 1
            info["events"].extend(reasons)
            info["reason"] = "+".join(reasons)
            return score, info

        trial.capacity_used += 1
        trial.enrolled_age_groups[patient.age_group] += 1
        self._assigned += 1

        reward = self._assignment_reward(patient, trial)
        info["events"].append("assigned")
        info["reason"] = "assignment_success"
        return _clamp(reward), info

    def _assignment_reward(self, patient: Patient, trial: Trial) -> float:
        reward = 0.40
        reward += 0.18 if patient.severity in trial.required_severity else 0.0
        reward += 0.18 if not any(comorbidity in trial.excluded_comorbidities for comorbidity in patient.comorbidities) else 0.0
        reward += 0.12 * (1.0 - trial.fill_rate)
        reward += 0.08 * ((trial.priority - 1) / 2.0)
        reward += 0.10 * self._diversity_bonus(patient, trial)
        reward -= min(0.01 * patient.wait_steps, 0.08)
        return _clamp(reward)

    def _assess_trial(self, patient: Patient, trial: Trial) -> tuple[bool, list[str], float]:
        reasons: list[str] = []
        if trial.is_full:
            reasons.append("trial_full")
        if patient.condition != trial.condition:
            reasons.append("condition_mismatch")
        if patient.severity not in trial.required_severity:
            reasons.append("severity_mismatch")

        bad_comorbidities = [item for item in patient.comorbidities if item in trial.excluded_comorbidities]
        if bad_comorbidities:
            reasons.append("comorbidity_mismatch")

        valid = not reasons
        if not valid:
            return False, reasons, 0.0

        reward = self._assignment_reward(patient, trial)
        return True, [], reward

    def _recommendation(self, patient: Patient) -> tuple[int, str, list[dict]]:
        trial_statuses = []
        best_trial_id = 0
        best_score = -1.0

        for trial in self.trials:
            valid, reasons, score = self._assess_trial(patient, trial)
            trial_statuses.append(
                {
                    "trial_id": trial.trial_id,
                    "valid": valid,
                    "reason": "valid" if valid else "+".join(reasons),
                    "score": float(round(score, 3)),
                }
            )
            if valid and score > best_score:
                best_score = score
                best_trial_id = trial.trial_id

        if best_trial_id == 0:
            return 0, "No valid trial available. Reject recommended.", trial_statuses

        return best_trial_id, f"Recommended trial {best_trial_id} with score {best_score:.2f}", trial_statuses

    def _diversity_bonus(self, patient: Patient, trial: Trial) -> float:
        if not trial.diversity_target:
            return 0.0
        target = trial.diversity_target.get(patient.age_group, 0.0)
        current = (
            trial.enrolled_age_groups[patient.age_group] / trial.capacity_used
            if trial.capacity_used > 0
            else 0.0
        )
        return max(0.0, target - current)

    def _end_of_episode_bonus(self) -> float:
        avg_fill = sum(trial.fill_rate for trial in self.trials) / max(len(self.trials), 1)
        accept_rate = self._assigned / max(len(self.patient_queue), 1)
        div_score = self._compute_diversity_index()
        bonus = 0.05 * avg_fill + 0.05 * accept_rate + 0.05 * div_score
        return _clamp(bonus)

    def _eligible_trials(self, patient: Patient) -> list[Trial]:
        return [
            trial
            for trial in self.trials
            if not trial.is_full
            and trial.condition == patient.condition
            and patient.severity in trial.required_severity
            and not any(comorbidity in trial.excluded_comorbidities for comorbidity in patient.comorbidities)
        ]

    def _compute_diversity_index(self) -> float:
        counts = {"pediatric": 0, "adult": 0, "elderly": 0}
        total = 0
        for trial in self.trials:
            for age_group, count in trial.enrolled_age_groups.items():
                counts[age_group] += count
                total += count
        if total == 0:
            return 0.0
        index = 0.0
        for count in counts.values():
            if count > 0:
                proportion = count / total
                index -= proportion * math.log(proportion + 1e-9)
        return min(1.0, index / math.log(len(counts)))

    def _advance_wait_times(self) -> None:
        for patient in self.patient_queue[self._step + 1 :]:
            patient.wait_steps += 1

    def _generate_trials(self) -> list[Trial]:
        trials = []
        conditions = self._rng.choices(CONDITIONS, k=self.n_trials)
        for trial_id, condition in enumerate(conditions, start=1):
            capacity = self._rng.randint(*self.capacity_range)
            severities = sorted(self._rng.sample([1, 2, 3], k=self._rng.randint(1, 3)))
            excluded = (
                self._rng.sample(COMORBIDS, k=self._rng.randint(0, 2))
                if self.difficulty != "easy"
                else []
            )
            diversity_target = {"elderly": 0.3, "pediatric": 0.15} if self.difficulty == "hard" else {}
            trials.append(
                Trial(
                    trial_id=trial_id,
                    condition=condition,
                    required_severity=severities,
                    excluded_comorbidities=excluded,
                    capacity_total=capacity,
                    priority=self._rng.randint(1, 3),
                    diversity_target=diversity_target,
                )
            )
        return trials

    def _generate_patients(self) -> list[Patient]:
        patients = []
        for index in range(self.n_patients):
            comorbidities = (
                self._rng.sample(COMORBIDS, k=self._rng.randint(0, 2))
                if self.difficulty != "easy"
                else []
            )
            patients.append(
                Patient(
                    patient_id=f"P{index + 1:04d}",
                    condition=self._rng.choice(CONDITIONS),
                    severity=self._rng.randint(1, 3),
                    age_group=self._rng.choice(AGE_GROUPS),
                    gender=self._rng.choice(GENDERS),
                    comorbidities=comorbidities,
                )
            )
        return patients
