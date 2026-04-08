#!/usr/bin/env python3
"""
Validation test suite for OpenEnv compliance.
Tests all graders and ensures scores are in valid range [0.0, 1.0]
"""
import sys
from env import ClinicalTrialEnv
from agent import RuleBasedAgent, GreedyFairnessAgent, RandomAgent, QLearningAgent
from tasks import TASK_MAP
from graders import grade_reward, grade_diversity, grade_assignment


def test_graders():
    """Test all three graders with different agents and tasks."""
    print("=" * 70)
    print("OPENENV GRADER VALIDATION TEST")
    print("=" * 70)

    test_cases = [
        ("medium", RuleBasedAgent(), "Rule-based"),
        ("easy", RandomAgent(4), "Random"),
        ("hard", GreedyFairnessAgent(), "Greedy Fairness"),
    ]

    all_valid = True

    for task, agent, agent_name in test_cases:
        print(f"\nTesting {agent_name} on {task.upper()} task...")
        print("-" * 70)

        config = dict(TASK_MAP[task])
        env = ClinicalTrialEnv(config)

        # Test Task 1: Reward Optimization
        print(f"  Task 1 - Reward Optimization: ", end="", flush=True)
        reward_result = grade_reward(env, agent, episodes=5)
        score = reward_result["score"]
        valid = 0.0 <= score <= 1.0
        all_valid = all_valid and valid
        status = "✓" if valid else "✗"
        print(f"{status} Score={score:.4f}")
        if not valid:
            print(f"    ERROR: Score out of range!")

        # Test Task 2: Fairness Optimization
        print(f"  Task 2 - Fairness Optimization: ", end="", flush=True)
        diversity_result = grade_diversity(env, agent, episodes=5)
        score = diversity_result["score"]
        valid = 0.0 <= score <= 1.0
        all_valid = all_valid and valid
        status = "✓" if valid else "✗"
        print(f"{status} Score={score:.4f}")
        if not valid:
            print(f"    ERROR: Score out of range!")

        # Test Task 3: Balanced Assignment
        print(f"  Task 3 - Balanced Assignment: ", end="", flush=True)
        assignment_result = grade_assignment(env, agent, episodes=5)
        score = assignment_result["score"]
        valid = 0.0 <= score <= 1.0
        all_valid = all_valid and valid
        status = "✓" if valid else "✗"
        print(f"{status} Score={score:.4f}")
        if not valid:
            print(f"    ERROR: Score out of range!")

    print("\n" + "=" * 70)
    if all_valid:
        print("✓ ALL TESTS PASSED - All scores in valid range [0.0, 1.0]")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME TESTS FAILED - Check error messages above")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(test_graders())
