from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Optional

CONDITIONS = ["oncology", "cardiology", "neurology", "immunology"]
AGE_GROUPS = ["pediatric", "adult", "elderly"]
GENDERS    = ["M", "F", "other"]
COMORBIDS  = ["diabetes", "hypertension", "renal_failure", "liver_disease"]


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

    def __init__(self, config: dict):
        self.n_patients      = config.get("n_patients", 50)
        self.n_trials        = config.get("n_trials", 6)
        self.capacity_range  = config.get("capacity_range", (8, 15))
        self.difficulty      = config.get("difficulty", "medium")
        self.fairness_weight = config.get("fairness_weight", 0.1)
        self.seed            = config.get("seed", 0)
        self._rng            = random.Random(self.seed)
        self.trials: list[Trial] = []
        self.patient_queue: list[Patient] = []
        self._step     = 0
        self._assigned = 0
        self._rejected = 0
        self._done     = False
        self._history: list[dict] = []

    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self._rng = random.Random(seed)
        self.trials        = self._generate_trials()
        self.patient_queue = self._generate_patients()
        self._step     = 0
        self._assigned = 0
        self._rejected = 0
        self._done     = False
        self._history  = []
        return {"observation": self.state(), "info": {"seed": seed or self.seed}}

    def step(self, action: int) -> dict:
        assert not self._done, "Episode finished — call reset()"
        assert 0 <= action <= self.n_trials, f"Invalid action {action}"

        patient = self.patient_queue[self._step]
        reward, info = self._compute_reward_and_apply(patient, action)

        self._history.append({
            "step":       self._step,
            "patient_id": patient.patient_id,
            "condition":  patient.condition,
            "severity":   patient.severity,
            "action":     action,
            "reward":     round(reward, 4),
            "events":     info.get("events", []),
        })

        self._step += 1
        terminated = (
            self._step >= len(self.patient_queue)
            or all(t.is_full for t in self.trials)
        )
        if terminated:
            self._done = True
            eob = self._end_of_episode_bonus()
            reward += eob
            info["end_of_episode_bonus"] = round(eob, 4)

        return {
            "observation": self.state(),
            "reward":      round(reward, 4),
            "terminated":  terminated,
            "truncated":   False,
            "info":        info,
        }

    def state(self) -> dict:
        if self._step >= len(self.patient_queue):
            current_patient = None
        else:
            p = self.patient_queue[self._step]
            current_patient = {
                "patient_id":      p.patient_id,
                "condition":       p.condition,
                "severity":        p.severity,
                "age_group":       p.age_group,
                "gender":          p.gender,
                "comorbidities":   p.comorbidities,
                "wait_steps":      p.wait_steps,
                "prior_rejection": p.prior_rejection,
            }

        return {
            "patient": current_patient,
            "trials": [
                {
                    "trial_id":               t.trial_id,
                    "condition":              t.condition,
                    "required_severity":      t.required_severity,
                    "excluded_comorbidities": t.excluded_comorbidities,
                    "capacity_total":         t.capacity_total,
                    "capacity_used":          t.capacity_used,
                    "slots_remaining":        t.slots_remaining,
                    "is_full":                t.is_full,
                    "fill_rate":              round(t.fill_rate, 3),
                    "priority":               t.priority,
                    "diversity_target":       t.diversity_target,
                }
                for t in self.trials
            ],
            "system": {
                "step":              self._step,
                "total_patients":    len(self.patient_queue),
                "patients_assigned": self._assigned,
                "patients_rejected": self._rejected,
                "diversity_index":   round(self._compute_diversity_index(), 3),
                "done":              self._done,
            },
        }

    def _compute_reward_and_apply(self, patient: Patient, action: int):
        reward = 0.0
        info   = {"action": action, "patient_id": patient.patient_id, "events": []}

        if action == 0:
            eligible = self._eligible_trials(patient)
            if eligible:
                reward -= 0.2  # Reduced penalty
                info["events"].append("unjustified_rejection")
            reward -= min(0.02 * patient.wait_steps, 0.3)  # Reduced wait penalty
            self._rejected += 1
            return reward, info

        trial = next((t for t in self.trials if t.trial_id == action), None)
        if trial is None:
            return -0.5, {"events": ["invalid_trial"]}  # Reduced penalty

        if trial.is_full:
            reward -= 0.8  # Reduced penalty
            info["events"].append("overflow_penalty")
            return reward, info

        if patient.condition != trial.condition:
            reward -= 1.0  # Reduced penalty
            info["events"].append("condition_mismatch")
            return reward, info

        # Base reward for successful assignment
        reward += 1.0

        reward += 1.0 * trial.priority  # Increased
        info["events"].append(f"condition_match +{1.0*trial.priority:.2f}")

        total_criteria = 1 + len(trial.excluded_comorbidities)
        criteria_met   = 0

        if patient.severity in trial.required_severity:
            criteria_met += 1
        else:
            reward -= 0.2  # Reduced
            info["events"].append("severity_violation")

        bad = [c for c in patient.comorbidities if c in trial.excluded_comorbidities]
        if not bad:
            criteria_met += len(trial.excluded_comorbidities)
        else:
            reward -= 0.2 * len(bad)  # Reduced
            info["events"].append(f"comorbidity_violations:{len(bad)}")

        reward += 1.0 * (criteria_met / max(total_criteria, 1))  # Increased

        div_bonus = self._diversity_bonus(patient, trial)
        reward   += div_bonus
        if div_bonus > 0:
            info["events"].append(f"diversity_bonus +{div_bonus:.2f}")

        reward -= 0.1 * trial.fill_rate  # Reduced
        reward -= min(0.02 * patient.wait_steps, 0.3)  # Reduced

        trial.capacity_used += 1
        trial.enrolled_age_groups[patient.age_group] += 1
        self._assigned += 1
        return reward, info

    def _diversity_bonus(self, patient: Patient, trial: Trial) -> float:
        if not trial.diversity_target:
            return 0.0
        target  = trial.diversity_target.get(patient.age_group, 0.0)
        current = (trial.enrolled_age_groups[patient.age_group] / trial.capacity_used
                   if trial.capacity_used > 0 else 0.0)
        return 1.0 * max(0.0, target - current)  # Increased

    def _end_of_episode_bonus(self) -> float:
        avg_fill    = sum(t.fill_rate for t in self.trials) / len(self.trials)
        accept_rate = self._assigned / max(len(self.patient_queue), 1)
        div_score   = self._compute_diversity_index()
        return round(2.0 * avg_fill + 1.0 * accept_rate + 1.0 * div_score, 3)

    def _eligible_trials(self, patient: Patient) -> list:
        return [
            t for t in self.trials
            if not t.is_full
            and t.condition == patient.condition
            and patient.severity in t.required_severity
            and not any(c in t.excluded_comorbidities for c in patient.comorbidities)
        ]

    def _compute_diversity_index(self) -> float:
        counts = {"pediatric": 0, "adult": 0, "elderly": 0}
        total  = 0
        for t in self.trials:
            for ag, n in t.enrolled_age_groups.items():
                counts[ag] += n
                total      += n
        if total == 0:
            return 0.0
        index = 0.0
        for n in counts.values():
            if n > 0:
                p      = n / total
                index -= p * math.log(p + 1e-9)
        return min(1.0, index / math.log(len(counts)))

    def _generate_trials(self) -> list:
        trials     = []
        conditions = self._rng.choices(CONDITIONS, k=self.n_trials)
        for i, cond in enumerate(conditions, start=1):
            cap  = self._rng.randint(*self.capacity_range)
            sevs = sorted(self._rng.sample([1, 2, 3], k=self._rng.randint(1, 3)))
            excl = (self._rng.sample(COMORBIDS, k=self._rng.randint(0, 2))
                    if self.difficulty != "easy" else [])
            div  = ({"elderly": 0.3, "pediatric": 0.15}
                    if self.difficulty == "hard" else {})
            trials.append(Trial(
                trial_id=i,
                condition=cond,
                required_severity=sevs,
                excluded_comorbidities=excl,
                capacity_total=cap,
                priority=self._rng.randint(1, 3),
                diversity_target=div,
            ))
        return trials

    def _generate_patients(self) -> list:
        patients = []
        for i in range(self.n_patients):
            comorbids = (self._rng.sample(COMORBIDS, k=self._rng.randint(0, 2))
                         if self.difficulty != "easy" else [])
            patients.append(Patient(
                patient_id=f"P{i+1:04d}",
                condition=self._rng.choice(CONDITIONS),
                severity=self._rng.randint(1, 3),
                age_group=self._rng.choice(AGE_GROUPS),
                gender=self._rng.choice(GENDERS),
                comorbidities=comorbids,
            ))
        return patients
