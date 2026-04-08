EASY_CONFIG = {
    "n_patients":     20,
    "n_trials":        4,
    "capacity_range": (8, 15),
    "difficulty":     "easy",
    "fairness_weight": 0.0,
    "seed":           42,
}

MEDIUM_CONFIG = {
    "n_patients":     50,
    "n_trials":        6,
    "capacity_range": (6, 12),
    "difficulty":     "medium",
    "fairness_weight": 0.1,
    "seed":           42,
}

HARD_CONFIG = {
    "n_patients":      100,
    "n_trials":          8,
    "capacity_range": (5, 10),
    "difficulty":     "hard",
    "fairness_weight": 0.3,
    "seed":           42,
}

TASK_MAP = {
    "easy":   EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard":   HARD_CONFIG,
}
