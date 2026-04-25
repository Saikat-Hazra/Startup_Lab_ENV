"""
Demo runner for startup simulation behavior tracking.

Runs 2 startups for 15 steps and prints:
- Step number
- Actions
- Rewards
- Insights generated

Also prints "Strategy change detected" when behavior changes.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.controller_agent import ControllerAgent
from env.startup_env import StartupEnv
from memory.episodic_store import EpisodicMemory
from memory.reflection import Reflection


def build_agent_state(env: StartupEnv, startup_idx: int) -> Dict[str, float]:
    """Build per-startup state for controller decisions."""
    startup = env.startup_states[startup_idx]
    return {
        "market_demand": float(env.shared_state["market_demand"]),
        "competition": float(env.shared_state["competition"]),
        "cash": float(startup["cash"]),
        "product_quality": float(startup["product_quality"]),
        "units_sold": float(startup["units_sold"]),
        "price": float(startup["price"]),
    }


def run_demo() -> None:
    """Run 2 startups for 15 steps and print behavior traces."""
    np.random.seed(7)

    env = StartupEnv(num_startups=2, max_steps=15, seed=7)
    controllers = [
        ControllerAgent(agent_id="startup_1", epsilon=0.15),
        ControllerAgent(agent_id="startup_2", epsilon=0.15),
    ]
    memory = EpisodicMemory(max_size=1000)
    reflection = Reflection(min_evidence=2)
    histories: List[List[Dict[str, float]]] = [[], []]

    env.reset()
    previous_behavior: Tuple[Tuple[int, ...], Tuple[Tuple[str, str], ...]] | None = None

    print("=" * 72)
    print("DEMO RUN (2 startups, 15 steps)")
    print("=" * 72)

    for step in range(15):
        states = [build_agent_state(env, i) for i in range(2)]
        actions: List[int] = []
        strategy_signature: List[Tuple[str, str]] = []

        for i, controller in enumerate(controllers):
            action, reasoning = controller.select_action(states[i], histories[i])
            actions.append(action)
            strategy = controller.decision_log[-1]["strategy"]
            strategy_signature.append(
                (strategy.get("primary_goal", "unknown"), strategy.get("risk_level", "unknown"))
            )

        _, rewards, done, _ = env.step(actions)
        next_states = [build_agent_state(env, i) for i in range(2)]

        for i in range(2):
            histories[i].append({"action": actions[i], "reward": rewards[i]})
            memory.add_experience(states[i], actions[i], rewards[i], next_states[i])

        insights = reflection.generate_insights(memory)
        for controller in controllers:
            controller.update_strategy(insights)

        action_names = [env.ACTIONS[a] for a in actions]
        print(f"\nStep {step + 1}/15")
        print(f"Actions: {action_names}")
        print(f"Rewards: {[round(r, 2) for r in rewards]}")
        print(f"Insights generated: {len(insights)}")
        if insights:
            print(f"Top insight: {insights[0]}")

        current_behavior = (tuple(actions), tuple(strategy_signature))
        if previous_behavior is not None and current_behavior != previous_behavior:
            print("Strategy change detected")
        previous_behavior = current_behavior

        if done:
            print(f"\nEpisode ended early at step {step + 1}")
            break

    print("\nDemo run complete.")


if __name__ == "__main__":
    run_demo()
