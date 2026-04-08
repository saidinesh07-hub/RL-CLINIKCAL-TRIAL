# 🚀 Quick Start Guide - OpenEnv Submission

## 1. Run Inference Immediately

```bash
# Simple: 10 episodes on medium difficulty with rule-based agent
python inference.py

# With parameters
python inference.py medium rule_based 10
python inference.py hard greedy_fairness 5
python inference.py easy random 3

# With environment variables
export API_BASE_URL=https://api.example.com
export MODEL_NAME=my-rl-model
python inference.py medium rule_based 10
```

## 2. Validate Everything Works

```bash
# Check all graders return valid scores
python validate_graders.py

# Full OpenEnv compliance check
python validate_openenv.py
```

## 3. Run with Docker

```bash
# Build Docker image
docker build -t clinical-trial-rl .

# Run Gradio UI (port 7860)
docker run -p 7860:7860 clinical-trial-rl

# Run inference.py
docker run --entrypoint "python inference.py medium rule_based 5" clinical-trial-rl

# With environment variables
docker run \
  -e API_BASE_URL=https://api.example.com \
  -e MODEL_NAME=my-model \
  --entrypoint "python inference.py medium rule_based 10" \
  clinical-trial-rl
```

## 4. API Endpoints

```bash
# Start API server
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# Reset environment
curl -X POST http://localhost:8000/reset?seed=42

# Step environment
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": 1}'

# Get state
curl http://localhost:8000/state

# Legacy simulation endpoint
curl http://localhost:8000/api/run-simulation
```

## 5. Available Agents

- `random` - Uniformly random baseline (lower bound)
- `rule_based` - Priority-aware greedy agent (good baseline)
- `greedy_fairness` - Fairness-aware agent (diversity optimized)
- `q_learning` - Q-learning agent (learnable)

## 6. Available Tasks

- `easy` - 20 patients, 4 trials, low difficulty
- `medium` - 50 patients, 6 trials, medium difficulty
- `hard` - 100 patients, 8 trials, high difficulty

## 7. Score Interpretation

All scores are in range **[0.0, 1.0]**:

| Score | Quality |
|-------|---------|
| 0.0-0.3 | Poor |
| 0.3-0.6 | Fair |
| 0.6-0.8 | Good |
| 0.8-1.0 | Excellent |

## 8. Output Format

```text
[START]
[STEP] Episode 1 → reward=+122.71, assignment=0.5200, diversity=0.9540
[STEP] Episode 2 → reward=+92.92, assignment=0.4600, diversity=0.9880
...
[END] Final Score: 0.7615
```

## 9. Three Competition Tasks

### Task 1: Reward Optimization
```python
python inference.py medium rule_based 10
# Scored via: graders.grade_reward()
# Measures: Maximized cumulative episode rewards
```

### Task 2: Fairness Optimization
```python
python inference.py hard greedy_fairness 10
# Scored via: graders.grade_diversity()
# Measures: Maintained diversity > 0.8
```

### Task 3: Balanced Assignment
```python
python inference.py medium rule_based 10
# Scored via: graders.grade_assignment()
# Measures: Stable assignment rate + fill rate
```

## 10. For Debugging

```bash
# Run with verbose output
python inference.py medium rule_based 5

# Test just the graders
python validate_graders.py

# Full compliance check
python validate_openenv.py

# Run training (with plots)
python main.py medium rule_based 200
```

## 11. Key Files

| File | Purpose |
|------|---------|
| `inference.py` | Main entry point for inference |
| `graders.py` | Task-specific graders |
| `openenv.yaml` | Task definitions and API spec |
| `validate_graders.py` | Grader validation test |
| `validate_openenv.py` | OpenEnv compliance checker |
| `env.py` | RL environment (unchanged) |
| `agent.py` | RL agents (unchanged) |
| `api.py` | FastAPI endpoints |
| `app.py` | Gradio UI (unchanged) |

## 12. Environment Variables

```bash
export API_BASE_URL=http://localhost:8000     # API base URL
export MODEL_NAME=clinical-trial-rl            # Model name
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx       # HF token
```

## 13. Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'inference'`
- **Solution**: Run from project root: `cd "c:\projects\clinical_trial_rl\RL AGENT"`

**Issue**: Matplotlib figure not found
- **Solution**: Inference.py uses Agg backend (no display needed) - this is normal

**Issue**: Docker build fails
- **Solution**: Ensure all dependencies in `requirements.txt` are correct

**Issue**: Score is always 1.0
- **Solution**: This is expected for rule-based agent on easy tasks - good baseline!

---

**Everything is ready for submission!** 🎉
