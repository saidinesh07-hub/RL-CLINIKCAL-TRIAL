from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from env import ClinicalTrialEnv
from tasks import TASK_MAP

from pydantic import BaseModel, Field


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


@app.get("/")
def root() -> HTMLResponse:
        return HTMLResponse(
                """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Clinical Trial RL Dashboard</title>
    <style>
        :root {
            --bg: #091323;
            --bg2: #142c4a;
            --card: #10203a;
            --accent: #4fd1c5;
            --accent2: #f6ad55;
            --text: #edf2f7;
            --muted: #a0aec0;
            --ok: #68d391;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            color: var(--text);
            background: radial-gradient(circle at top right, var(--bg2), var(--bg));
            min-height: 100vh;
            padding: 24px;
        }
        .wrap { max-width: 1000px; margin: 0 auto; }
        .title { margin: 0 0 6px; font-size: 30px; }
        .sub { margin: 0 0 16px; color: var(--muted); }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 12px;
            margin-bottom: 12px;
        }
        .card {
            background: var(--card);
            border: 1px solid #27456f;
            border-radius: 12px;
            padding: 12px;
        }
        label { display: block; color: var(--muted); font-size: 13px; margin-bottom: 6px; }
        input, select {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #335b8f;
            background: #0b1b30;
            color: var(--text);
        }
        .actions { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 14px; }
        button {
            border: 0;
            border-radius: 9px;
            padding: 10px 12px;
            background: var(--accent);
            color: #06241f;
            font-weight: 700;
            cursor: pointer;
        }
        button.secondary { background: var(--accent2); color: #2a1900; }
        button.ghost { background: #26476f; color: var(--text); }
        .metrics { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }
        .pill {
            background: #0b1b30;
            border: 1px solid #335b8f;
            border-radius: 999px;
            padding: 6px 10px;
            color: var(--ok);
            font-size: 13px;
        }
        pre {
            margin: 0;
            background: #071425;
            border: 1px solid #27456f;
            border-radius: 10px;
            padding: 12px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 360px;
            overflow: auto;
        }
    </style>
</head>
<body>
    <div class="wrap">
        <h1 class="title">Clinical Trial RL Dashboard</h1>
        <p class="sub">Visible UI for the OpenEnv API. Use controls below to reset and step the environment.</p>

        <div class="grid">
            <div class="card">
                <label for="task">Task</label>
                <select id="task">
                    <option value="easy">easy</option>
                    <option value="medium" selected>medium</option>
                    <option value="hard">hard</option>
                </select>
            </div>
            <div class="card">
                <label for="seed">Seed</label>
                <input id="seed" type="number" value="0" />
            </div>
            <div class="card">
                <label for="action">Action (0 reject, 1..N trial id)</label>
                <input id="action" type="number" min="0" value="1" />
            </div>
        </div>

        <div class="actions">
            <button id="btn-reset">Reset</button>
            <button id="btn-step" class="secondary">Step</button>
            <button id="btn-recommended" class="ghost">Step Recommended</button>
            <button id="btn-state" class="ghost">Get State</button>
        </div>

        <div class="metrics" id="metrics"></div>

        <div class="card">
            <pre id="output">Ready.</pre>
        </div>
    </div>

    <script>
        const output = document.getElementById('output');
        const metrics = document.getElementById('metrics');

        function show(data) {
            output.textContent = JSON.stringify(data, null, 2);
            const obs = (data || {}).observation || {};
            const sys = obs.system || {};
            const trialCount = Array.isArray(obs.trials) ? obs.trials.length : 0;
            const parts = [
                `step: ${sys.step ?? '-'}`,
                `assigned: ${sys.patients_assigned ?? '-'}`,
                `rejected: ${sys.patients_rejected ?? '-'}`,
                `diversity: ${Number(sys.diversity_index ?? 0).toFixed(3)}`,
                `fill: ${Number(sys.average_fill_rate ?? 0).toFixed(3)}`,
                `recommended: ${sys.recommended_action ?? '-'}`,
                `trials: ${trialCount}`,
                `done: ${sys.done ?? data.done ?? false}`
            ];
            metrics.innerHTML = parts.map(v => `<span class="pill">${v}</span>`).join('');
        }

        async function postJson(path, payload) {
            const res = await fetch(path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            return await res.json();
        }

        async function getJson(path) {
            const res = await fetch(path);
            return await res.json();
        }

        document.getElementById('btn-reset').onclick = async () => {
            const task = document.getElementById('task').value;
            const seed = Number(document.getElementById('seed').value);
            show(await postJson('/reset', { task, seed }));
        };

        document.getElementById('btn-step').onclick = async () => {
            const action = Number(document.getElementById('action').value);
            show(await postJson('/step', { action }));
        };

        document.getElementById('btn-recommended').onclick = async () => {
            const s = await getJson('/state');
            const rec = Number((((s || {}).observation || {}).system || {}).recommended_action || 0);
            document.getElementById('action').value = String(rec);
            show(await postJson('/step', { action: rec }));
        };

        document.getElementById('btn-state').onclick = async () => {
            show(await getJson('/state'));
        };

        getJson('/state').then(show).catch(err => {
            output.textContent = `Failed to load state: ${err}`;
        });
    </script>
</body>
</html>
"""
        )


@app.get("/api")
def api_info() -> dict[str, Any]:
        return {
                "status": "ok",
                "service": "clinical-trial-openenv",
                "endpoints": ["/reset", "/step", "/state", "/health"],
        }


@app.post("/reset")
@app.post("/reset/")
async def reset_env(request: Request) -> dict[str, Any]:
    global _env

    try:
        raw_body = await request.body()
        if not raw_body:
            data = {}
        else:
            parsed = await request.json()
            data = parsed if isinstance(parsed, dict) else {}
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

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
