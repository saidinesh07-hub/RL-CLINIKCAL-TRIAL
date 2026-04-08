#!/usr/bin/env python3
"""
OpenEnv Compliance Validator
Checks that all required components exist and work correctly
"""
import sys
import inspect
from env import ClinicalTrialEnv
from tasks import TASK_MAP
from graders import grade_reward, grade_diversity, grade_assignment
import yaml

def validate_env_methods():
    """Check that required environment methods exist."""
    print("\n[1] Validating Environment Methods...")
    print("-" * 60)
    
    env = ClinicalTrialEnv(TASK_MAP["medium"])
    
    required_methods = ["reset", "step", "state"]
    all_exist = True
    
    for method in required_methods:
        exists = hasattr(env, method) and callable(getattr(env, method))
        status = "✓" if exists else "✗"
        print(f"  {status} {method}() - {'OK' if exists else 'MISSING'}")
        all_exist = all_exist and exists
    
    # Test that they actually work
    print("\n  Testing method execution...")
    try:
        result = env.reset(seed=0)
        assert "observation" in result
        print(f"    ✓ reset() returns proper observation")
    except Exception as e:
        print(f"    ✗ reset() failed: {e}")
        all_exist = False
    
    try:
        result = env.step(0)
        assert all(k in result for k in ["observation", "reward", "terminated"])
        print(f"    ✓ step() returns proper result dict")
    except Exception as e:
        print(f"    ✗ step() failed: {e}")
        all_exist = False
    
    try:
        state = env.state()
        assert "patient" in state and "trials" in state and "system" in state
        print(f"    ✓ state() returns proper observation dict")
    except Exception as e:
        print(f"    ✗ state() failed: {e}")
        all_exist = False
    
    return all_exist


def validate_tasks():
    """Check that all 3 tasks are defined."""
    print("\n[2] Validating Task Definitions...")
    print("-" * 60)
    
    with open("openenv.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    tasks = config.get("tasks", [])
    required_task_ids = ["task_1_reward_optimization", "task_2_fairness_optimization", "task_3_balanced_assignment"]
    
    all_exist = True
    for task_id in required_task_ids:
        task_exists = any(t["id"] == task_id for t in tasks)
        status = "✓" if task_exists else "✗"
        print(f"  {status} {task_id} - {'OK' if task_exists else 'MISSING'}")
        all_exist = all_exist and task_exists
    
    return all_exist


def validate_graders():
    """Check that all graders exist and return valid scores."""
    print("\n[3] Validating Task Graders...")
    print("-" * 60)
    
    from agent import RuleBasedAgent
    
    graders_list = [
        ("grade_reward", grade_reward),
        ("grade_diversity", grade_diversity),
        ("grade_assignment", grade_assignment),
    ]
    
    all_valid = True
    for name, grader_func in graders_list:
        exists = callable(grader_func)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}() - {'OK' if exists else 'MISSING'}")
        
        if exists:
            try:
                env = ClinicalTrialEnv(TASK_MAP["medium"])
                agent = RuleBasedAgent()
                result = grader_func(env, agent, episodes=2)
                score = result.get("score")
                in_range = 0.0 <= score <= 1.0 if score is not None else False
                score_status = "✓" if in_range else "✗"
                print(f"      {score_status} Returns score in [0.0, 1.0]: {score:.4f}")
                all_valid = all_valid and in_range
            except Exception as e:
                print(f"      ✗ Error calling grader: {e}")
                all_valid = False
    
    return all_valid


def validate_formatting():
    """Check that inference.py output format is correct."""
    print("\n[4] Validating Output Format...")
    print("-" * 60)
    
    # Import and test directly
    try:
        import inference
        from agent import RuleBasedAgent
        
        # Capture output from function
        env = ClinicalTrialEnv(TASK_MAP["easy"])
        agent = RuleBasedAgent()
        results = inference.run_inference("easy", "random", 2)
        
        score = results.get("score")
        episodes = results.get("episodes", [])
        
        all_valid = True
        
        # Check that we got episodes
        if len(episodes) >= 2:
            print(f"  ✓ {len(episodes)} episodes generated")
        else:
            print(f"  ✗ Missing episodes (found {len(episodes)})")
            all_valid = False
        
        # Check score is in range
        if 0.0 <= score <= 1.0:
            print(f"  ✓ Final score in [0.0, 1.0]: {score:.4f}")
        else:
            print(f"  ✗ Final score out of range: {score:.4f}")
            all_valid = False
        
        # Check that each episode has proper metrics
        if episodes:
            ep = episodes[0]
            required_keys = ["episode", "reward", "assignment_rate", "diversity_score"]
            has_keys = all(k in ep for k in required_keys)
            if has_keys:
                print(f"  ✓ Episode data has required fields")
            else:
                print(f"  ✗ Episode data missing required fields")
                all_valid = False
        
        return all_valid
    except Exception as e:
        print(f"  ✗ Error during format validation: {e}")
        return False


def validate_endpoints():
    """Check that API endpoints exist in openenv.yaml."""
    print("\n[5] Validating API Endpoints...")
    print("-" * 60)
    
    with open("openenv.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    endpoints = config.get("endpoints", {})
    required_endpoints = ["reset", "step", "state"]
    
    all_exist = True
    for endpoint in required_endpoints:
        exists = endpoint in endpoints
        status = "✓" if exists else "✗"
        print(f"  {status} {endpoint} endpoint - {'OK' if exists else 'MISSING'}")
        all_exist = all_exist and exists
    
    return all_exist


def main():
    print("=" * 60)
    print("OPENENV COMPLIANCE VALIDATOR")
    print("=" * 60)
    
    results = []
    results.append(("Environment Methods", validate_env_methods()))
    results.append(("Task Definitions", validate_tasks()))
    results.append(("Task Graders", validate_graders()))
    results.append(("Output Format", validate_formatting()))
    results.append(("API Endpoints", validate_endpoints()))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_valid = True
    for name, valid in results:
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"  {status} - {name}")
        all_valid = all_valid and valid
    
    print("=" * 60)
    
    if all_valid:
        print("✓ ALL VALIDATIONS PASSED - Project is OpenEnv compliant!")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED - See errors above")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
