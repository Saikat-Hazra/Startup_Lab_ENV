"""Integration loop: agent + memory + reflection + environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.controller_agent import ControllerAgent
from agents.validator import DecisionValidator
from env.startup_env import StartupEnv
from memory.episodic_store import EpisodicMemory
from memory.reflection import Reflection


def run_simulation(steps: int = 25) -> None:
    env = StartupEnv(max_steps=steps, seed=42)
    agent = ControllerAgent()
    validator = DecisionValidator()
    memory = EpisodicMemory(max_size=1000)
    reflection = Reflection()
    insights = []
    history = []

    state = env.reset()
    print("Starting simulation...")

    for step in range(steps):
        actions = []
        for _ in range(env.num_startups):
            action, reasoning = agent.select_action(state, insights)
            action = validator.validate(state, action, history)
            if action not in agent.allowed_actions:
                action, reasoning = agent.refine_action(state, insights)
                action = validator.validate(state, action, history)
            actions.append(action)

        next_state, rewards, done = env.step(actions)

        # Store experience
        for i, action in enumerate(actions):
            memory.add_experience(state, action, rewards[i])
            history.append({"state": state, "action": action, "reward": rewards[i]})

        # Every 5 steps -> generate new insights
        if (step + 1) % 5 == 0:
            insights = reflection.analyze(memory.get_recent(100))

        print(
            f"Step {step + 1:02d} | actions={actions} | rewards="
            f"{[round(r, 3) for r in rewards]} | insights={len(insights)}"
        )

        state = next_state
        if done:
            print("Simulation ended early.")
            break

    print("Simulation complete.")


if __name__ == "__main__":
    run_simulation(steps=25)
