"""
Demo script for startup simulation with strategy-aware controllers.

Runs 2 startups for 10 steps and prints:
- Actions taken
- Rewards received
- Strategies used

Also reports early-vs-late reward averages to show behavior improvement.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.controller_agent import ControllerAgent
from env.startup_env import StartupEnv
from memory.episodic_store import EpisodicMemory
from memory.reflection import Reflection


def build_agent_state(env: StartupEnv, startup_idx: int) -> Dict[str, float]:
    """Build per-startup state dict for controller decisions."""
    startup = env.startup_states[startup_idx]
    return {
        "market_demand": float(env.shared_state["market_demand"]),
        "competition": float(env.shared_state["competition"]),
        "cash": float(startup["cash"]),
        "product_quality": float(startup["product_quality"]),
        "units_sold": float(startup["units_sold"]),
        "price": float(startup["price"]),
    }


def summarize_improvement(step_rewards: List[List[float]]) -> None:
    """Print early-vs-late reward comparison for each startup."""
    print("\n" + "=" * 72)
    print("IMPROVEMENT SUMMARY")
    print("=" * 72)

    for i, rewards in enumerate(step_rewards):
        split = len(rewards) // 2
        early_avg = np.mean(rewards[:split]) if split > 0 else 0.0
        late_avg = np.mean(rewards[split:]) if rewards[split:] else 0.0
        delta = late_avg - early_avg
        trend = "improved" if delta >= 0 else "declined"
        print(
            f"Startup {i + 1}: early avg={early_avg:.2f}, "
            f"late avg={late_avg:.2f}, delta={delta:+.2f} ({trend})"
        )

    total_per_step = [sum(step_pair) for step_pair in zip(*step_rewards)]
    split_total = len(total_per_step) // 2
    total_early = np.mean(total_per_step[:split_total]) if split_total > 0 else 0.0
    total_late = np.mean(total_per_step[split_total:]) if total_per_step[split_total:] else 0.0
    total_delta = total_late - total_early
    print(
        f"Combined system: early avg={total_early:.2f}, "
        f"late avg={total_late:.2f}, delta={total_delta:+.2f}"
    )


def run_demo() -> None:
    """Run a 2-startup, 10-step demonstration."""
    np.random.seed(42)

    env = StartupEnv(num_startups=2, max_steps=10, seed=42)
    controllers = [
        ControllerAgent(agent_id="startup_1", epsilon=0.15),
        ControllerAgent(agent_id="startup_2", epsilon=0.15),
    ]
    memory = EpisodicMemory(max_size=1000)
    reflection = Reflection(min_evidence=2)

    histories: List[List[Dict[str, float]]] = [[], []]
    step_rewards: List[List[float]] = [[], []]

    env.reset()

    print("=" * 72)
    print("STARTUP SIMULATION DEMO (2 startups, 10 steps)")
    print("=" * 72)

    for step in range(10):
        states = [build_agent_state(env, i) for i in range(2)]

        actions = []
        strategies = []
        for i, controller in enumerate(controllers):
            action = controller.select_action(states[i], histories[i])
            actions.append(action)
            strategies.append(controller.decision_log[-1]["strategy"])

        _, rewards, done, info = env.step(actions)
        next_states = [build_agent_state(env, i) for i in range(2)]

        print(f"\nStep {step + 1}/10")
        print("-" * 72)
        for i in range(2):
            action_name = env.ACTIONS[actions[i]]
            strategy = strategies[i]
            print(
                f"Startup {i + 1} | "
                f"Action: {action_name:<15} | "
                f"Reward: {rewards[i]:>7.2f} | "
                f"Strategy: goal={strategy['primary_goal']}, risk={strategy['risk_level']}"
            )

            histories[i].append({"action": actions[i], "reward": rewards[i]})
            step_rewards[i].append(rewards[i])
            memory.add_experience(states[i], actions[i], rewards[i], next_states[i])

        # Reflect periodically and update controller strategies
        if (step + 1) % 5 == 0:
            insights = reflection.generate_insights(memory)
            for controller in controllers:
                controller.update_strategy(insights)

            print("\nReflection update:")
            print(f"- Insights generated: {len(insights)}")
            print(f"- Top insight: {insights[0]}")

        print(
            f"Actions this step: {info['actions']} | "
            f"Rewards: {[round(r, 2) for r in rewards]}"
        )

        if done:
            print(f"\nEpisode ended early at step {step + 1}")
            break

    summarize_improvement(step_rewards)
    print("\nDemo complete.")


if __name__ == "__main__":
    run_demo()
