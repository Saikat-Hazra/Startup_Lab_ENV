"""FastAPI backend for startup simulation."""

import os
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.controller_agent import ControllerAgent
from agents.validator import DecisionValidator
from env.startup_env import StartupEnv
from memory.episodic_store import EpisodicMemory
from memory.reflection import Reflection


class StepRequest(BaseModel):
    actions: List[str] | None = None


app = FastAPI(title="Self-Improving Autonomous Startup Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation components
env = StartupEnv(max_steps=50, seed=42)
agent = ControllerAgent()
validator = DecisionValidator()
memory = EpisodicMemory(max_size=2000)
reflection = Reflection()
last_insights: List[str] = []
history: List[Dict[str, Any]] = []
USE_LLM_AGENT = os.getenv("USE_LLM_AGENT", "false").lower() == "true"


def _auto_actions(state: Dict[str, Any]) -> List[str]:
    if not USE_LLM_AGENT:
        return ["analyze_market"] * env.num_startups

    actions = []
    for _ in range(env.num_startups):
        try:
            candidate = agent.select_action(state, last_insights)
            valid = validator.validate(state, candidate, history)
            if valid != candidate:
                valid = agent.refine_action(state, last_insights)
                valid = validator.validate(state, valid, history)
            actions.append(valid)
        except Exception:
            # Safety fallback so /step never returns 500 for model-side issues.
            actions.append("analyze_market")
    return actions


@app.get("/state")
def get_state() -> Dict[str, Any]:
    return {
        "state": env.get_state(),
        "actions": [],
        "rewards": [],
        "insights": last_insights,
    }


@app.post("/step")
def run_step(payload: StepRequest) -> Dict[str, Any]:
    global last_insights

    state_before = env.get_state()
    actions = payload.actions if payload.actions else _auto_actions(state_before)
    actions = [
        validator.validate(state_before, action, history)
        for action in actions[: env.num_startups]
    ]
    while len(actions) < env.num_startups:
        actions.append("analyze_market")

    next_state, rewards, done = env.step(actions)

    for i in range(env.num_startups):
        memory.add_experience(state_before, actions[i], rewards[i])
        history.append(
            {
                "state": state_before,
                "action": actions[i],
                "reward": rewards[i],
            }
        )

    if env.current_step % 5 == 0:
        last_insights = reflection.analyze(memory.get_recent(100))

    return {
        "state": next_state,
        "actions": actions,
        "rewards": rewards,
        "insights": last_insights,
        "done": done,
    }


@app.get("/logs")
def get_logs() -> Dict[str, Any]:
    return {
        "state": env.get_state(),
        "actions": [entry["actions"] for entry in env.logs],
        "rewards": [entry["rewards"] for entry in env.logs],
        "insights": last_insights,
        "logs": env.logs,
    }
