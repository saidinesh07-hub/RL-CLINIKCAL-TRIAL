import random


class RandomAgent:
    """Uniformly random baseline — lower bound for performance."""

    def __init__(self, n_trials: int, seed: int = 99):
        self.n_trials = n_trials
        self._rng     = random.Random(seed)

    def act(self, observation: dict) -> int:
        return self._rng.randint(0, self.n_trials)

    @property
    def name(self) -> str:
        return "Random agent"


class RuleBasedAgent:
    """
    Priority-aware greedy agent.
    1. Find trials matching condition, not full, no eligibility violations.
    2. Sort by (highest priority, lowest fill rate).
    3. Reject only when no eligible trial exists.
    """

    def act(self, observation: dict) -> int:
        patient = observation.get("patient")
        if patient is None:
            return 0

        eligible = []
        for trial in observation["trials"]:
            if trial["is_full"]:
                continue
            if trial["condition"] != patient["condition"]:
                continue
            if patient["severity"] not in trial["required_severity"]:
                continue
            if any(c in trial["excluded_comorbidities"]
                   for c in patient["comorbidities"]):
                continue
            eligible.append(trial)

        if not eligible:
            return 0

        eligible.sort(key=lambda t: (-t["priority"], t["fill_rate"]))
        return eligible[0]["trial_id"]

    @property
    def name(self) -> str:
        return "Rule-based agent"


class GreedyFairnessAgent:
    """
    Extends RuleBasedAgent with diversity awareness.
    Prefers trials where the patient's age group is under-represented
    relative to the trial's diversity target.
    """

    def act(self, observation: dict) -> int:
        patient = observation.get("patient")
        if patient is None:
            return 0

        eligible = []
        for trial in observation["trials"]:
            if trial["is_full"]:
                continue
            if trial["condition"] != patient["condition"]:
                continue
            if patient["severity"] not in trial["required_severity"]:
                continue
            if any(c in trial["excluded_comorbidities"]
                   for c in patient["comorbidities"]):
                continue

            div_target = trial["diversity_target"].get(patient["age_group"], 0.0)
            enrolled   = trial["capacity_used"]
            eligible.append((trial, div_target, enrolled))

        if not eligible:
            return 0

        # Score: priority + diversity gap bonus − fill penalty
        def score(item):
            t, div_target, enrolled = item
            div_gap = max(0.0, div_target - (enrolled / max(t["capacity_total"], 1)))
            return t["priority"] + 0.5 * div_gap - 0.3 * t["fill_rate"]

        eligible.sort(key=score, reverse=True)
        return eligible[0][0]["trial_id"]

    @property
    def name(self) -> str:
        return "Greedy-fairness agent"


class QLearningAgent:
    """Q-learning agent for clinical trial assignment with epsilon decay."""

    def __init__(self, n_trials: int, alpha: float = 0.1, gamma: float = 0.9, epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: float = 0.995, seed: int = 42):
        self.n_trials = n_trials
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._rng = random.Random(seed)
        self.q_table = {}  # state -> list of q_values for actions 0 to n_trials
        self.episode_count = 0

    def _get_state(self, observation: dict) -> tuple:
        patient = observation.get("patient")
        if patient is None:
            return ("no_patient",)
        # Simple state: condition, severity, age_group, sorted comorbidities tuple
        return (patient["condition"], patient["severity"], patient["age_group"], tuple(sorted(patient["comorbidities"])))

    def act(self, observation: dict) -> int:
        state = self._get_state(observation)
        if state not in self.q_table:
            self.q_table[state] = [0.0] * (self.n_trials + 1)
        
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.n_trials)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values)
            # Random among max
            candidates = [i for i, q in enumerate(q_values) if q == max_q]
            return self._rng.choice(candidates)

    def learn(self, observation: dict, action: int, reward: float, next_observation: dict, done: bool):
        state = self._get_state(observation)
        next_state = self._get_state(next_observation)
        
        if state not in self.q_table:
            self.q_table[state] = [0.0] * (self.n_trials + 1)
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * (self.n_trials + 1)
        
        q_values = self.q_table[state]
        next_q_values = self.q_table[next_state]
        
        target = reward + (0 if done else self.gamma * max(next_q_values))
        q_values[action] += self.alpha * (target - q_values[action])

    def update_epsilon(self):
        self.episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @property
    def name(self) -> str:
        return "Q-learning agent"
