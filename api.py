from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import json
import os
from env import ClinicalTrialEnv
from agent import RuleBasedAgent
from tasks import TASK_MAP
from graders import grade

# Environment variable support
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "clinical-trial-rl")
HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance for API sessions
_env: ClinicalTrialEnv | None = None
_agent = None
_step_count = 0


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.post("/reset")
async def reset_env(seed: int = 42):
    """Reset environment and return initial observation."""
    global _env, _agent, _step_count
    config = dict(TASK_MAP["medium"])
    _env = ClinicalTrialEnv(config)
    _agent = RuleBasedAgent()
    _step_count = 0
    result = _env.reset(seed=seed)
    return {
        "observation": result["observation"],
        "info": result["info"],
    }


@app.post("/step")
async def step_env(action: int):
    """Execute action in environment."""
    global _env, _step_count
    if _env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    result = _env.step(action)
    _step_count += 1
    return {
        "observation": result["observation"],
        "reward": result["reward"],
        "terminated": result["terminated"],
        "truncated": result["truncated"],
        "info": result["info"],
    }


@app.get("/state")
async def get_state():
    """Get current environment state."""
    global _env
    if _env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    return _env.state()


@app.get("/api/run-simulation")
async def run_simulation():
    """Legacy endpoint - run simulation and return episode data."""
    # Run the Python training script
    result = subprocess.run([
        "python", "main.py", "medium", "q_learning", "200"
    ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

    # Parse the output to extract episode data
    episodes = []
    final_metrics = {}

    lines = result.stdout.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('Episode ') and ': Reward=' in line:
            # Parse episode line: "Episode 1: Reward=+5.20, Assignment Rate=0.300, Diversity=0.850"
            parts = line.split(': ')
            ep_part = parts[0].split()[1]  # '1'
            metrics_part = parts[1]  # 'Reward=+5.20, Assignment Rate=0.300, Diversity=0.850'
            metrics = {}
            for item in metrics_part.split(', '):
                key, value = item.split('=')
                metrics[key.lower().replace(' ', '')] = float(value)
            episodes.append({
                "episode": int(ep_part),
                "reward": metrics['reward'],
                "assignmentRate": metrics['assignmentrate'],
                "diversityScore": metrics['diversity']
            })
        elif 'Overall score' in line:
            # Parse evaluation: "    Overall score   : 0.8500 / 1.0000"
            final_metrics['score'] = float(line.split(':')[1].split('/')[0].strip())
        elif 'Assignment rate' in line and 'test episodes' not in line:
            final_metrics['assignmentRate'] = float(line.split(':')[1].strip())
        elif 'Diversity index' in line:
            final_metrics['diversity'] = float(line.split(':')[1].strip())
        elif 'Fill rate' in line:
            final_metrics['fillRate'] = float(line.split(':')[1].strip())
        elif 'Mean reward' in line:
            final_metrics['reward'] = float(line.split(':')[1].strip())

    # Return the parsed episodes data
    return episodes
