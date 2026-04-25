"""
PPO Training Script using HuggingFace TRL.

Minimal PPO implementation for multi-agent startup simulation.
Uses simple transformer policy, collects rollouts, and optimizes with PPO.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from collections import deque

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from env.startup_env import StartupEnv
from training.config import TrainingConfig


class SimplePolicy(nn.Module):
    """Simple neural network policy for action selection."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize policy network.

        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            hidden_size: Hidden layer size
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        self.value_head = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns logits and value.

        Args:
            state: State tensor [batch_size, state_size]

        Returns:
            logits: Action logits
            value: State value estimate
        """
        logits = self.net(state)
        value = self.value_head(state)
        return logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value from state.

        Args:
            state: State tensor

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: Value estimate
        """
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)


class RolloutCollector:
    """Collect rollouts from environment interactions."""

    def __init__(
        self,
        env: StartupEnv,
        policy: SimplePolicy,
        device: str = "cpu",
    ):
        """
        Initialize rollout collector.

        Args:
            env: StartupEnv instance
            policy: Policy network
            device: Torch device
        """
        self.env = env
        self.policy = policy
        self.device = device

    def collect_rollout(self, num_steps: int = 100) -> Dict[str, Any]:
        """
        Collect one rollout from environment.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Dictionary containing states, actions, rewards, log_probs, values
        """
        states = []
        actions_list = []
        rewards_list = []
        log_probs_list = []
        values_list = []
        dones = []

        state = self.env.reset()

        for step in range(num_steps):
            # Convert state to tensor
            state_tensor = torch.from_numpy(state).float().to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.policy.get_action_and_value(
                    state_tensor.unsqueeze(0)
                )

            # Convert actions to list (one per startup)
            action_list = action.cpu().numpy().tolist()

            # Execute step in environment
            next_state, rewards, done, info = self.env.step(action_list)

            # Store trajectory
            states.append(state.copy())
            actions_list.append(action_list)
            rewards_list.append(rewards)  # List of rewards per startup
            log_probs_list.append(log_prob.cpu().numpy())
            values_list.append(value.cpu().numpy())
            dones.append(done)

            state = next_state

            if done:
                break

        return {
            "states": np.array(states),
            "actions": np.array(actions_list),
            "rewards": np.array(rewards_list),  # [steps, num_startups]
            "log_probs": np.array(log_probs_list),
            "values": np.array(values_list),
            "dones": np.array(dones),
            "episode_length": step + 1,
        }


class PPOTrainer:
    """PPO training manager."""

    def __init__(
        self,
        env: StartupEnv,
        policy: SimplePolicy,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize PPO trainer.

        Args:
            env: StartupEnv instance
            policy: Policy network
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: Lambda for GAE
            clip_ratio: PPO clip ratio
            entropy_coeff: Entropy coefficient
            value_coeff: Value function loss coefficient
            device: Torch device
        """
        self.env = env
        self.policy = policy
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff

        self.optimizer = Adam(policy.parameters(), lr=learning_rate)
        self.collector = RolloutCollector(env, policy, device)

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def compute_returns_and_advantages(
        self,
        rollout: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and GAE advantages.

        Args:
            rollout: Rollout dictionary

        Returns:
            returns: Computed returns
            advantages: Computed advantages
        """
        rewards = rollout["rewards"]  # [steps, num_startups]
        values = rollout["values"]  # [steps, num_startups] or [steps]
        dones = rollout["dones"]

        # Average rewards across startups for training simplicity
        avg_rewards = rewards.mean(axis=1)  # [steps]

        # If values is 1D, keep it; if 2D, average
        if values.ndim == 2:
            avg_values = values.mean(axis=1)
        else:
            avg_values = values

        # Compute GAE
        advantages = np.zeros_like(avg_rewards)
        returns = np.zeros_like(avg_rewards)

        last_gae = 0.0
        for t in reversed(range(len(avg_rewards))):
            if t == len(avg_rewards) - 1:
                next_value = 0.0
            else:
                next_value = avg_values[t + 1]

            if dones[t]:
                delta = avg_rewards[t] - avg_values[t]
            else:
                delta = avg_rewards[t] + self.gamma * next_value - avg_values[t]

            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

            returns[t] = advantages[t] + avg_values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def ppo_update(self, rollout: Dict[str, Any], num_epochs: int = 4) -> Dict[str, float]:
        """
        Perform PPO update on collected rollout.

        Args:
            rollout: Rollout dictionary
            num_epochs: Number of gradient descent epochs

        Returns:
            Training metrics
        """
        returns, advantages = self.compute_returns_and_advantages(rollout)

        states = torch.from_numpy(rollout["states"]).float().to(self.device)
        old_log_probs = torch.from_numpy(rollout["log_probs"]).float().to(self.device)
        returns_tensor = torch.from_numpy(returns).float().to(self.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
        }

        for epoch in range(num_epochs):
            # Forward pass
            logits, values = self.policy(states)
            values = values.squeeze(-1)

            # Compute policy loss with PPO clipping
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(
                torch.from_numpy(rollout["actions"].mean(axis=1)).long().to(self.device)
            )

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                * advantages_tensor
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = 0.5 * ((values - returns_tensor) ** 2).mean()

            # Entropy bonus
            entropy = dist.entropy().mean()
            entropy_loss = -self.entropy_coeff * entropy

            # Total loss
            total_loss = policy_loss + self.value_coeff * value_loss + entropy_loss

            # Gradient step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            metrics["policy_loss"] = policy_loss.item()
            metrics["value_loss"] = value_loss.item()
            metrics["entropy_loss"] = entropy_loss.item()
            metrics["total_loss"] = total_loss.item()

        return metrics

    def train(
        self,
        num_iterations: int = 10,
        rollout_steps: int = 100,
        log_interval: int = 5,
    ) -> Dict[str, Any]:
        """
        Run PPO training loop.

        Args:
            num_iterations: Number of rollout + update iterations
            rollout_steps: Steps per rollout
            log_interval: Log every N iterations

        Returns:
            Training results
        """
        print(f"Starting PPO training for {num_iterations} iterations...")
        print(f"Rollout steps: {rollout_steps}")
        print(f"Startups: {self.env.num_startups}\n")

        for iteration in range(num_iterations):
            # Collect rollout
            rollout = self.collector.collect_rollout(num_steps=rollout_steps)

            # Compute rewards
            rewards = rollout["rewards"]  # [steps, num_startups]
            avg_reward = rewards.mean()
            total_reward = rewards.sum()

            self.episode_rewards.append(float(total_reward))
            self.episode_lengths.append(rollout["episode_length"])

            # PPO update
            metrics = self.ppo_update(rollout)

            # Logging
            if (iteration + 1) % log_interval == 0:
                avg_ep_reward = np.mean(self.episode_rewards[-log_interval:])
                avg_ep_length = np.mean(self.episode_lengths[-log_interval:])

                print(
                    f"Iteration {iteration + 1}/{num_iterations} | "
                    f"Reward: {avg_ep_reward:.2f} | "
                    f"Length: {avg_ep_length:.1f} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f}"
                )

        print("\nTraining complete!")

        return {
            "num_iterations": num_iterations,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "avg_reward": float(np.mean(self.episode_rewards)),
            "max_reward": float(np.max(self.episode_rewards)),
            "final_reward": float(self.episode_rewards[-1]),
        }

    def plot_rewards(self, output_dir: Path = None) -> None:
        """
        Plot training rewards per iteration and save as PNG.

        Creates visualization with:
        - Iteration rewards line plot
        - Moving average overlay
        - Clear axis labels and title

        Args:
            output_dir: Directory to save plot (default: current working directory)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available, skipping reward plot")
            return

        if not self.episode_rewards:
            print("No episode rewards to plot")
            return

        if output_dir is None:
            output_dir = Path.cwd()

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot episode rewards
        iterations = np.arange(1, len(self.episode_rewards) + 1)
        ax.plot(
            iterations,
            self.episode_rewards,
            label="Iteration Reward",
            color="steelblue",
            alpha=0.7,
            linewidth=1.5,
        )

        # Plot moving average (window=5)
        if len(self.episode_rewards) >= 5:
            moving_avg = np.convolve(
                self.episode_rewards, np.ones(5) / 5, mode="valid"
            )
            ax.plot(
                iterations[4:],
                moving_avg,
                label="5-Iteration Moving Average",
                color="coral",
                linewidth=2.5,
            )

        # Labels and title
        ax.set_xlabel("Iteration", fontsize=12, fontweight="bold")
        ax.set_ylabel("Total Reward", fontsize=12, fontweight="bold")
        ax.set_title("PPO Training Rewards Over Iterations", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc="best")

        # Add statistics text box
        stats_text = (
            f"Max Reward: {np.max(self.episode_rewards):.2f}\n"
            f"Avg Reward: {np.mean(self.episode_rewards):.2f}\n"
            f"Final Reward: {self.episode_rewards[-1]:.2f}"
        )
        ax.text(
            0.98,
            0.05,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Save plot
        plot_file = output_dir / "ppo_reward_plot.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        print(f"\nReward plot saved to {plot_file}")
        plt.close()


def main():
    """Main training entry point."""
    # Configuration
    config = TrainingConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Environment
    env = StartupEnv(num_startups=config.num_agents, max_steps=config.max_steps)
    print(f"Environment: {env.num_startups} startups, {config.max_steps} max steps")
    print(f"State size: {env.observation_space.shape[0]}, Action space: {env.action_space}\n")

    # Policy
    state_size = env.observation_space.shape[0]
    action_size = 5  # 5 possible actions per startup
    policy = SimplePolicy(state_size, action_size, hidden_size=128).to(device)
    print(f"Policy: {state_size} -> 128 -> {action_size}\n")

    # Trainer
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        device=device,
    )

    # Train
    results = trainer.train(
        num_iterations=config.num_episodes,
        rollout_steps=config.max_steps,
        log_interval=5,
    )

    # Print results
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print(f"Final Reward:       {results['final_reward']:.2f}")
    print(f"Average Reward:     {results['avg_reward']:.2f}")
    print(f"Max Reward:         {results['max_reward']:.2f}")

    # Save results
    results_file = Path(config.output_dir) / "ppo_results.json"
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Save policy
    policy_file = Path(config.output_dir) / "ppo_policy.pt"
    torch.save(policy.state_dict(), policy_file)
    print(f"Policy saved to {policy_file}")

    # Plot rewards
    trainer.plot_rewards(output_dir=Path(config.output_dir))


if __name__ == "__main__":
    main()
