"""
Multi-startup simulation environment.

Includes Gym-style reset/step/state methods and optional OpenEnv base compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np

try:
    # Optional dependency: available in validator/runtime that has OpenEnv installed.
    from openenv.core.env_server.interfaces import Environment as _OpenEnvEnvironment
except Exception:  # pragma: no cover - local fallback when openenv is unavailable
    class _OpenEnvEnvironment:  # type: ignore[no-redef]
        """Fallback base so local development works without openenv installed."""


class _SimpleSpace:
    """Small placeholder object mimicking common Gym space attributes."""

    def __init__(self, shape: Optional[Tuple[int, ...]] = None, n: Optional[int] = None):
        self.shape = shape
        self.n = n

    def sample(self) -> int:
        if self.n is None:
            raise ValueError("Cannot sample from non-discrete space.")
        return int(np.random.randint(0, self.n))


class StartupEnv(_OpenEnvEnvironment):
    """Simple startup simulation with competing startups."""

    ACTIONS = [
        "build_product",
        "improve_quality",
        "run_marketing",
        "reduce_price",
        "analyze_market",
    ]

    def __init__(
        self,
        max_steps: int = 50,
        seed: Optional[int] = None,
        num_startups: int = 2,
    ):
        self.num_startups = max(1, int(num_startups))
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.logs: List[Dict[str, Any]] = []
        self.startups: List[Dict[str, float]] = []
        self.market_demand: float = 100.0
        self.observation_space = _SimpleSpace(shape=(self.num_startups * 2 + 2,))
        self.action_space = _SimpleSpace(n=len(self.ACTIONS))
        self.reset()

    @property
    def startup_states(self) -> List[Dict[str, float]]:
        """Backward-compatible alias used by legacy training code."""
        return self.startups

    @property
    def state(self) -> Dict[str, Any]:
        """OpenEnv-compatible state property."""
        return self.get_state()

    def reset(
        self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.market_demand = 100.0
        self.logs = []
        self.startups = []
        for _ in range(self.num_startups):
            self.startups.append({"cash": 100_000.0, "product_quality": 50.0})
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """Return current observable state."""
        return {
            "step": self.current_step,
            "market_demand": round(self.market_demand, 2),
            "startups": copy.deepcopy(self.startups),
        }

    def _normalize_action(self, action: Any) -> str:
        if isinstance(action, int):
            if 0 <= action < len(self.ACTIONS):
                return self.ACTIONS[action]
            return "analyze_market"
        if isinstance(action, str) and action in self.ACTIONS:
            return action
        return "analyze_market"

    def _state_to_vector(self, state: Dict[str, Any]) -> np.ndarray:
        startups = state.get("startups", [])
        vec: List[float] = [float(state.get("market_demand", 0.0)), float(state.get("step", 0))]
        for i in range(self.num_startups):
            startup = startups[i] if i < len(startups) else {}
            vec.append(float(startup.get("cash", 0.0)))
            vec.append(float(startup.get("product_quality", 0.0)))
        return np.array(vec, dtype=np.float32)

    def step(
        self, actions: List[Any], timeout_s: Optional[float] = None, **kwargs: Any
    ) -> Tuple[Dict[str, Any], List[float], bool, Dict[str, Any]]:
        """
        Execute one step for two startups.

        Returns:
            next_state, rewards, done
        """
        if len(actions) != self.num_startups:
            raise ValueError(f"Expected exactly {self.num_startups} actions, one per startup.")

        self.current_step += 1
        costs = [0.0, 0.0]

        # Apply actions
        normalized_actions: List[str] = []
        for i, raw_action in enumerate(actions):
            action = self._normalize_action(raw_action)
            normalized_actions.append(action)

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

        # Competition: higher quality gets larger demand share.
        quality_values = [max(1.0, s["product_quality"]) for s in self.startups]
        total_q = float(sum(quality_values))
        shares = [q / total_q for q in quality_values]

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

        total_profit = float(sum(revenues) - sum(costs))
        self.logs.append(
            {
                "step": self.current_step,
                "actions": normalized_actions,
                "rewards": [round(r, 3) for r in rewards],
                "market_demand": round(self.market_demand, 2),
                "qualities": [s["product_quality"] for s in self.startups],
                "cash": [round(s["cash"], 2) for s in self.startups],
            }
        )
        info = {
            "profits": [float(r - c) for r, c in zip(revenues, costs)],
            "total_profit": total_profit,
            "state_vector": self._state_to_vector(self.get_state()).tolist(),
        }
        return self.get_state(), rewards, done, info
