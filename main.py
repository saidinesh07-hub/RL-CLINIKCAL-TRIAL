from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env import ClinicalTrialEnv
from tasks import TASK_MAP


class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=None, description="Optional reset seed")
    task: str = Field(default="medium", description="Task difficulty: easy|medium|hard")


class StepRequest(BaseModel):
    action: int = Field(..., ge=0, description="0 reject, 1..N trial id")


app = FastAPI(title="Clinical Trial OpenEnv API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: ClinicalTrialEnv | None = None


def _build_env(task: str) -> ClinicalTrialEnv:
    cfg = dict(TASK_MAP.get(task, TASK_MAP["medium"]))
    return ClinicalTrialEnv(cfg)


def _to_api_state(observation: dict[str, Any]) -> list[float]:
    patient = observation.get("patient", {}) or {}
    system = observation.get("system", {}) or {}
    trials = observation.get("trials", []) or []

    total_patients = max(int(system.get("total_patients", 0)), 1)
    n_trials = max(len(trials), 1)

    state = [
        float(int(system.get("step", 0))),
        float(int(system.get("patients_assigned", 0)) / total_patients),
        float(int(system.get("patients_rejected", 0)) / total_patients),
        float(int(system.get("invalid_actions", 0)) / total_patients),
        float(system.get("diversity_index", 0.0)),
        float(system.get("average_fill_rate", 0.0)),
        float(patient.get("severity", 0)),
        float(len(patient.get("comorbidities", []) or [])),
        float(int(system.get("recommended_action", 0)) / n_trials),
    ]
    return state


def _reset_response(observation: dict[str, Any], task: str) -> dict[str, Any]:
    return {
        "observation": observation,
        "state": _to_api_state(observation),
        "task": task,
        "reward": 0.0,
        "done": False,
        "terminated": False,
        "truncated": False,
        "info": {"reason": "reset"},
    }


def _step_response(result: dict[str, Any]) -> dict[str, Any]:
    observation = result.get("observation", {})
    reward = float(result.get("reward", 0.0))
    terminated = bool(result.get("terminated", False))
    truncated = bool(result.get("truncated", False))
    info = result.get("info", {}) or {}
    done = terminated or truncated

    return {
        "observation": observation,
        "state": _to_api_state(observation),
        "reward": reward,
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
async def reset_env(request: Request) -> dict[str, Any]:
    global _env

    try:
        data = await request.json()
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    seed = data.get("seed", None)
    task = data.get("task", "medium")
    if task not in TASK_MAP:
        task = "medium"

    _env = _build_env(task)
    result = _env.reset(seed=seed)
    observation = result.get("observation", _env.state())

    return _reset_response(observation, task)


@app.post("/step")
def step_env(payload: StepRequest) -> dict[str, Any]:
    global _env

    if _env is None:
        _env = _build_env("medium")
        result = _env.reset(seed=0)
        observation = result.get("observation", _env.state())
        return {
            **_reset_response(observation, "medium"),
            "info": {"reason": "auto_reset_before_step"},
        }

    result = _env.step(payload.action)
    return _step_response(result)


@app.get("/state")
def state_env() -> dict[str, Any]:
    global _env

    if _env is None:
        _env = _build_env("medium")
        result = _env.reset(seed=0)
        observation = result.get("observation", _env.state())
    else:
        observation = _env.state()

    done = bool((observation.get("system", {}) or {}).get("done", False))
    return {
        "observation": observation,
        "state": _to_api_state(observation),
        "done": done,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
