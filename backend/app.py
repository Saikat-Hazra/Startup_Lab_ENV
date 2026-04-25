"""FastAPI backend for startup simulation."""

import os
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.controller_agent import ControllerAgent
from agents.validator import DecisionValidator
from memory.episodic_store import EpisodicMemory
from memory.reflection import Reflection


def _load_startup_env_class():
    """Load StartupEnv with a file-path fallback for container runtimes."""
    try:
        from env.startup_env import StartupEnv as StartupEnvClass

        return StartupEnvClass
    except ModuleNotFoundError:
        env_file = Path(__file__).resolve().parent.parent / "env" / "startup_env.py"
        if env_file.exists():
            spec = importlib.util.spec_from_file_location("startup_env_local", env_file)
            if spec is None or spec.loader is None:
                raise
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.StartupEnv

        # Final fallback for constrained/containerized runtimes.
        import copy
        import random

        class StartupEnvClass:  # type: ignore[no-redef]
            ACTIONS = [
                "build_product",
                "improve_quality",
                "run_marketing",
                "reduce_price",
                "analyze_market",
            ]

            def __init__(self, max_steps: int = 50, seed: Optional[int] = None):
                self.num_startups = 2
                self.max_steps = max_steps
                self.rng = random.Random(seed)
                self.current_step = 0
                self.logs: List[Dict[str, Any]] = []
                self.startups: List[Dict[str, float]] = []
                self.market_demand = 100.0
                self.reset()

            def reset(self) -> Dict[str, Any]:
                self.current_step = 0
                self.market_demand = 100.0
                self.logs = []
                self.startups = [
                    {"cash": 100_000.0, "product_quality": 50.0},
                    {"cash": 100_000.0, "product_quality": 50.0},
                ]
                return self.get_state()

            def get_state(self) -> Dict[str, Any]:
                return {
                    "step": self.current_step,
                    "market_demand": round(self.market_demand, 2),
                    "startups": copy.deepcopy(self.startups),
                }

            def step(self, actions: List[str]):
                if len(actions) != self.num_startups:
                    raise ValueError("Expected exactly 2 actions, one per startup.")

                self.current_step += 1
                costs = [0.0, 0.0]
                for i, action in enumerate(actions):
                    if action not in self.ACTIONS:
                        action = "analyze_market"
                    if action == "build_product":
                        costs[i] = 7000.0
                        self.startups[i]["product_quality"] = min(
                            100.0, self.startups[i]["product_quality"] + 6.0
                        )
                    elif action == "improve_quality":
                        costs[i] = 4000.0
                        self.startups[i]["product_quality"] = min(
                            100.0, self.startups[i]["product_quality"] + 3.0
                        )
                    elif action == "run_marketing":
                        costs[i] = 5000.0
                        self.market_demand = min(200.0, self.market_demand + 8.0)
                    elif action == "reduce_price":
                        costs[i] = 1000.0
                    elif action == "analyze_market":
                        costs[i] = 500.0
                    self.startups[i]["cash"] -= costs[i]

                self.market_demand = max(
                    20.0, self.market_demand + self.rng.gauss(0.0, 2.5)
                )
                q1 = max(1.0, self.startups[0]["product_quality"])
                q2 = max(1.0, self.startups[1]["product_quality"])
                total_q = q1 + q2
                shares = [q1 / total_q, q2 / total_q]
                rewards = []
                for i, share in enumerate(shares):
                    revenue = share * self.market_demand * 120.0
                    self.startups[i]["cash"] += revenue
                    rewards.append(float((revenue - costs[i]) / 1000.0))

                done = self.current_step >= self.max_steps or any(
                    startup["cash"] <= 0 for startup in self.startups
                )
                self.logs.append(
                    {
                        "step": self.current_step,
                        "actions": actions,
                        "rewards": [round(r, 3) for r in rewards],
                        "market_demand": round(self.market_demand, 2),
                        "qualities": [s["product_quality"] for s in self.startups],
                        "cash": [round(s["cash"], 2) for s in self.startups],
                    }
                )
                return self.get_state(), rewards, done

        return StartupEnvClass


StartupEnv = _load_startup_env_class()


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

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"
if FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS)), name="assets")

# Global simulation components
env = StartupEnv(max_steps=50, seed=42)
validator = DecisionValidator()
memory = EpisodicMemory(max_size=2000)
reflection = Reflection()
last_insights: List[str] = []
history: List[Dict[str, Any]] = []
USE_LLM_AGENT = os.getenv("USE_LLM_AGENT", "false").lower() == "true"

try:
    agent = ControllerAgent() if USE_LLM_AGENT else None
except Exception:
    # Keep API available even when LLM provider/key is unavailable.
    USE_LLM_AGENT = False
    agent = None


def _generate_commentary(actions: List[str], rewards: List[float], state_before: Dict[str, Any], state_after: Dict[str, Any]) -> str:
    if not USE_LLM_AGENT or agent is None:
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
    if mode == "baseline" or not USE_LLM_AGENT or agent is None:
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


@app.get("/")
def index():
    """Serve built frontend in container, fallback to API info locally."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Startup Lab API is running",
        "docs": "/docs",
        "state_endpoint": "/state",
    }


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
