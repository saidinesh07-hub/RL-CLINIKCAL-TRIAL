from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os

from env import ClinicalTrialEnv
from agent import RuleBasedAgent
from tasks import TASK_MAP

# Environment variables
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

# Global state
_env = None
_agent = None


# ================= HEALTH =================
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


# ================= RESET (FIXED) =================
@app.post("/reset")
async def reset_env():
    global _env, _agent

    config = dict(TASK_MAP["medium"])
    _env = ClinicalTrialEnv(config)
    _agent = RuleBasedAgent()

    result = _env.reset()

    return {
        "observation": result.get("observation", {})
    }


# ================= STEP (FIXED - NO BODY REQUIRED) =================
@app.post("/step")
async def step_env():
    global _env

    if _env is None:
        return {
            "observation": {},
            "reward": 0.0,
            "terminated": True,
            "truncated": False,
            "info": {}
        }

    action = 0  # default (IMPORTANT)

    result = _env.step(action)

    return {
        "observation": result.get("observation", {}),
        "reward": float(result.get("reward", 0.0)),
        "terminated": bool(result.get("terminated", False)),
        "truncated": False,
        "info": {}
    }


# ================= OPTIONAL STATE =================
@app.get("/state")
async def get_state():
    global _env
    if _env is None:
        return {"error": "Environment not initialized"}
    return _env.state()


# ================= LEGACY SIMULATION =================
@app.get("/api/run-simulation")
async def run_simulation():
    result = subprocess.run(
        ["python", "main.py", "medium", "q_learning", "200"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__)
    )

    return {"output": result.stdout}