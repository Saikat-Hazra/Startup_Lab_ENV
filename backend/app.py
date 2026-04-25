"""FastAPI backend for startup simulation."""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.controller_agent import ControllerAgent
from agents.validator import DecisionValidator
from env.startup_env import StartupEnv
from memory.episodic_store import EpisodicMemory
from memory.reflection import Reflection


class StepRequest(BaseModel):
    actions: Optional[List[str]] = None
    mode: str = "trained"


app = FastAPI(title="Self-Improving Autonomous Startup Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "http://localhost:5178",
        "http://localhost:5179",
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


def _generate_commentary(actions: List[str], rewards: List[float], state_before: Dict[str, Any], state_after: Dict[str, Any]) -> str:
    if not USE_LLM_AGENT:
        return "Simulation step completed."

    prompt = f"""
You are a sports commentator for startup battles.

Actions taken: {actions}
Rewards: {rewards}
Previous state: {state_before}
Current state: {state_after}

Provide exciting commentary in 2-3 sentences like a sports announcer.
"""
    try:
        return agent.model.generate(prompt).strip()
    except:
        return "Exciting moves in the startup arena!"


def _auto_actions(state: Dict[str, Any], mode: str = "trained") -> tuple[List[str], List[str]]:
    if mode == "baseline" or not USE_LLM_AGENT:
        actions = ["analyze_market"] * env.num_startups
        reasonings = ["Baseline action"] * env.num_startups
        return actions, reasonings

    actions = []
    reasonings = []
    for _ in range(env.num_startups):
        try:
            action, reasoning = agent.select_action(state, last_insights)
            valid = validator.validate(state, action, history)
            if valid != action:
                action, reasoning = agent.refine_action(state, last_insights)
                valid = validator.validate(state, valid, history)
            actions.append(valid)
            reasonings.append(reasoning)
        except Exception:
            # Safety fallback so /step never returns 500 for model-side issues.
            actions.append("analyze_market")
            reasonings.append("Fallback due to error")
    return actions, reasonings


@app.get("/state")
def get_state() -> Dict[str, Any]:
    return {
        "state": env.get_state(),
        "actions": [],
        "rewards": [],
        "insights": last_insights,
        "reasonings": [],
        "commentary": "",
    }


@app.post("/step")
def run_step(payload: StepRequest) -> Dict[str, Any]:
    global last_insights

    state_before = env.get_state()
    if payload.actions:
        actions = payload.actions
        reasonings = ["Manual action"] * len(actions)
    else:
        actions, reasonings = _auto_actions(state_before, payload.mode)
    actions = [
        validator.validate(state_before, action, history)
        for action in actions[: env.num_startups]
    ]
    while len(actions) < env.num_startups:
        actions.append("analyze_market")
        reasonings.append("Default action")
    reasonings = reasonings[:env.num_startups]

    next_state, rewards, done = env.step(actions)

    for i in range(env.num_startups):
        memory.add_experience(state_before, actions[i], rewards[i])
        history.append(
            {
                "state": state_before,
                "action": actions[i],
                "reward": rewards[i],
                "reasoning": reasonings[i],
            }
        )

    if env.current_step % 5 == 0:
        last_insights = reflection.analyze(memory.get_recent(100))

    # Generate commentary
    commentary = _generate_commentary(actions, rewards, state_before, next_state)

    return {
        "state": next_state,
        "actions": actions,
        "rewards": rewards,
        "insights": last_insights,
        "reasonings": reasonings,
        "commentary": commentary,
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
