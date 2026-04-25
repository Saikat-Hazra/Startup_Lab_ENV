"""
OpenEnv-compatible inference runner for Startup Lab Env.

Outputs validator-friendly log lines:
- [START]
- [STEP]
- [END]
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import requests

from env.startup_env import StartupEnv

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASK_ID = os.getenv("TASK", "startup-lab-default")
MODEL_NAME = os.getenv("MODEL_NAME", "heuristic-policy")
BENCHMARK = "startup-lab-env"
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def env_post(path: str, payload: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
    url = f"{ENV_URL}{path}"
    response = requests.post(url, json=payload or {}, params=params or {}, timeout=20)
    response.raise_for_status()
    return response.json()


def choose_actions(observation: Dict) -> List[str]:
    startups = observation.get("startups", [])
    actions: List[str] = []
    for startup in startups:
        cash = float(startup.get("cash", 0.0))
        quality = float(startup.get("product_quality", 0.0))
        if cash < 20_000:
            actions.append("analyze_market")
        elif quality < 55:
            actions.append("improve_quality")
        else:
            actions.append("run_marketing")
    return actions or ["analyze_market"]


def run_episode() -> Dict[str, float]:
    total_reward = 0.0
    rewards: List[float] = []
    done = False
    step = 0
    success = False

    log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)
    local_env: Optional[StartupEnv] = None
    try:
        reset_resp = env_post("/reset")
        observation = reset_resp.get("state", reset_resp)
    except Exception:
        # Local fallback keeps script executable even when API server is not running.
        local_env = StartupEnv(max_steps=MAX_STEPS, seed=42)
        observation = local_env.reset()

    while not done and step < MAX_STEPS:
        step += 1
        error: Optional[str] = None
        actions = choose_actions(observation)
        action_str = json.dumps({"actions": actions})

        try:
            if local_env is None:
                step_resp = env_post("/step", payload={"actions": actions, "mode": "trained"})
                observation = step_resp.get("state", {})
                reward = float(sum(step_resp.get("rewards", [0.0])))
                done = bool(step_resp.get("done", False))
            else:
                observation, local_rewards, done, _info = local_env.step(actions)
                reward = float(sum(local_rewards))
        except Exception as exc:
            reward = 0.0
            done = True
            error = str(exc).replace("\n", " ")

        rewards.append(reward)
        total_reward += reward
        log_step(step=step, action=action_str, reward=reward, done=done, error=error)

    success = total_reward > 0
    score = max(0.0, min(1.0, total_reward / 1000.0))
    log_end(success=success, steps=step, score=score, rewards=rewards)
    return {"score": score, "steps": float(step), "success": float(success)}


def main() -> None:
    result = run_episode()
    print(
        f"[SUMMARY] score={result['score']:.4f} steps={int(result['steps'])} "
        f"success={bool(result['success'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
