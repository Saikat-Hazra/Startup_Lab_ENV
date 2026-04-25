"""
Multi-startup simulation environment.

Two startups compete for shared market demand. Better quality captures more demand.
"""

from typing import Any, Dict, List, Tuple
import copy
import numpy as np


class StartupEnv:
    """Simple startup simulation with 2 competing startups."""

    ACTIONS = [
        "build_product",
        "improve_quality",
        "run_marketing",
        "reduce_price",
        "analyze_market",
    ]

    def __init__(self, max_steps: int = 50, seed: int | None = None):
        self.num_startups = 2
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.logs: List[Dict[str, Any]] = []
        self.startups: List[Dict[str, float]] = []
        self.market_demand: float = 100.0
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.market_demand = 100.0
        self.logs = []
        self.startups = [
            {"cash": 100_000.0, "product_quality": 50.0},
            {"cash": 100_000.0, "product_quality": 50.0},
        ]
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """Return current observable state."""
        return {
            "step": self.current_step,
            "market_demand": round(self.market_demand, 2),
            "startups": copy.deepcopy(self.startups),
        }

    def step(self, actions: List[str]) -> Tuple[Dict[str, Any], List[float], bool]:
        """
        Execute one step for two startups.

        Returns:
            next_state, rewards, done
        """
        if len(actions) != self.num_startups:
            raise ValueError("Expected exactly 2 actions, one per startup.")

        self.current_step += 1
        costs = [0.0, 0.0]

        # Apply actions
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

        # Natural market movement
        self.market_demand = max(
            20.0, self.market_demand + float(self.rng.normal(loc=0.0, scale=2.5))
        )

        # Competition: higher quality gets larger demand share
        q1 = max(1.0, self.startups[0]["product_quality"])
        q2 = max(1.0, self.startups[1]["product_quality"])
        total_q = q1 + q2
        shares = [q1 / total_q, q2 / total_q]

        revenues = []
        rewards = []
        for i, share in enumerate(shares):
            revenue = share * self.market_demand * 120.0
            revenues.append(revenue)
            self.startups[i]["cash"] += revenue
            reward = (revenue - costs[i]) / 1000.0
            rewards.append(float(reward))

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
