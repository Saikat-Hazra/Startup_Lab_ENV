"""
Training script for startup simulation agents.

Implements DQN-based training loop for agents to learn optimal startup strategies.
Integrates episodic memory, reflection, and learning from experience.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from collections import deque
import os

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.startup_env import StartupEnv
from rewards.reward_function import RewardFunction
from training.config import TrainingConfig
from memory.episodic_store import EpisodicMemory
from memory.reflection import Reflection
from agents.controller_agent import ControllerAgent


class DQNAgent:
    """
    Simple DQN agent for startup environment.
    
    Uses experience replay and epsilon-greedy exploration.
    """
    
    def __init__(
        self,
        action_size: int,
        state_size: int,
        config: TrainingConfig,
        agent_id: str = "agent_0",
    ):
        """
        Initialize DQN agent.
        
        Args:
            action_size: Number of possible actions
            state_size: Size of state vector
            config: Training configuration
            agent_id: Unique agent identifier
        """
        self.agent_id = agent_id
        self.action_size = action_size
        self.state_size = state_size
        self.config = config
        
        # Q-learning parameters
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma
        
        # Simple Q-table (state discretization)
        # For simplicity, we use a dictionary-based Q-table
        self.q_table: Dict[str, np.ndarray] = {}
        
        # Experience replay
        self.memory = deque(maxlen=config.replay_buffer_size)
        self.reward_function = RewardFunction(
            business_weight=1.0,
            learning_weight=0.5,
            adaptation_weight=0.3,
            repetition_penalty_weight=-0.5,
        )
    
    def _state_to_key(self, state: Dict[str, float]) -> str:
        """
        Convert continuous state to discrete key for Q-table.
        
        Uses simple binning strategy.
        
        Args:
            state: State dictionary
        
        Returns:
            Discrete state key
        """
        # Discretize each state dimension
        cash_bin = int(min(9, state["cash"] / 100000))
        quality_bin = int(state["product_quality"] / 10)
        demand_bin = int(state["market_demand"] / 10)
        competition_bin = int(state["competition"] / 10)
        
        return f"({cash_bin},{quality_bin},{demand_bin},{competition_bin})"
    
    def _init_q_values(self, state_key: str) -> None:
        """Initialize Q-values for a new state."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
    
    def act(self, state: Dict[str, float], training: bool = True) -> int:
        """
        Select action using epsilon-greedy strategy.
        
        Args:
            state: Current state dictionary
            training: Whether in training mode (use exploration)
        
        Returns:
            Action index (0-4)
        """
        state_key = self._state_to_key(state)
        self._init_q_values(state_key)
        
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        return int(np.argmax(self.q_table[state_key]))
    
    def remember(
        self,
        state: Dict[str, float],
        action: int,
        reward: float,
        next_state: Dict[str, float],
        done: bool,
    ) -> None:
        """
        Store experience in replay memory.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode completion flag
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int) -> float:
        """
        Learn from experience replay batch.
        
        Args:
            batch_size: Number of experiences to replay
        
        Returns:
            Average loss
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        batch = [self.memory[i] for i in np.random.choice(len(self.memory), batch_size)]
        
        total_loss = 0.0
        for state, action, reward, next_state, done in batch:
            state_key = self._state_to_key(state)
            next_state_key = self._state_to_key(next_state)
            
            self._init_q_values(state_key)
            self._init_q_values(next_state_key)
            
            # Q-learning update
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.q_table[next_state_key])
            
            old_q = self.q_table[state_key][action]
            self.q_table[state_key][action] += self.learning_rate * (target - old_q)
            
            loss = (target - old_q) ** 2
            total_loss += loss
        
        return total_loss / batch_size if batch_size > 0 else 0.0
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str) -> None:
        """
        Save agent Q-table to file.
        
        Args:
            filepath: Path to save file
        """
        # Convert numpy arrays to lists for JSON serialization
        q_table_json = {k: v.tolist() for k, v in self.q_table.items()}
        
        data = {
            "agent_id": self.agent_id,
            "action_size": self.action_size,
            "state_size": self.state_size,
            "epsilon": self.epsilon,
            "q_table": q_table_json,
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Agent {self.agent_id} saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load agent Q-table from file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self.epsilon = data.get("epsilon", self.epsilon)
        q_table_json = data.get("q_table", {})
        self.q_table = {k: np.array(v) for k, v in q_table_json.items()}
        
        print(f"Agent {self.agent_id} loaded from {filepath}")


class Trainer:
    """
    Training manager for startup simulation agents.
    
    Handles episode loops, metric tracking, and model checkpointing.
    """
    
    def __init__(self, config: TrainingConfig, env: StartupEnv = None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            env: Optional pre-initialized environment
        """
        self.config = config
        self.env = env or StartupEnv(num_startups=config.num_agents, max_steps=config.max_steps)
        
        # Create agents
        self.agents = {}
        self.controller_agents = {}  # Memory-aware agents
        # For multi-startup: state_size = num_startups * 4 + 2 (cash, quality, units, price per startup + market_demand, competition)
        state_size = config.num_agents * 4 + 2
        for i in range(config.num_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = DQNAgent(
                action_size=5,
                state_size=state_size,
                config=config,
                agent_id=agent_id,
            )
            self.controller_agents[agent_id] = ControllerAgent(agent_id=agent_id)
        
        # Memory and reflection system
        self.episodic_memory = EpisodicMemory()
        self.reflection = Reflection(min_evidence=2)
        self.reward_function = RewardFunction()
        
        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_profits: List[float] = []
        self.episode_insights: List[int] = []  # Number of insights per episode
        self.episode_repeated_actions: List[int] = []  # Count of repeated actions
        self.best_reward = -float("inf")
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Dict[str, Any]:
        """
        Run training loop for all agents.
        
        Includes memory, reflection, and learning updates.
        
        Returns:
            Training results dictionary
        """
        print(f"Starting training for {self.config.num_episodes} episodes...")
        print(f"Number of agents: {self.config.num_agents}")
        print(f"Max steps per episode: {self.config.max_steps}\n")
        
        for episode in range(self.config.num_episodes):
            episode_reward = 0.0
            episode_length = 0
            episode_profit = 0.0
            episode_history: List[Dict[str, Any]] = []
            repeated_action_count = 0
            
            # Reset environment and all agents
            state = self.env.reset()
            for controller_agent in self.controller_agents.values():
                controller_agent.reset()
            
            # Episode loop - all agents act simultaneously
            for step in range(self.config.max_steps):
                # All agents select actions simultaneously
                actions = []
                for agent_id in self.agents.keys():
                    dqn_agent = self.agents[agent_id]
                    controller_agent = self.controller_agents[agent_id]

                    # Agent selects action via memory-aware controller
                    action = controller_agent.select_action(state, episode_history)
                    actions.append(action)

                # Execute step for all startups
                next_state, rewards, done, info = self.env.step(actions)

                # Process results for each agent
                for i, agent_id in enumerate(self.agents.keys()):
                    dqn_agent = self.agents[agent_id]
                    reward = rewards[i]

                    # Store in DQN replay memory using composite reward
                    dqn_agent.remember(state, actions[i], reward, next_state, done)

                    # 🧠 MEMORY STEP 1: Store experience in episodic memory
                    self.episodic_memory.add_experience(state, actions[i], reward, next_state)

                # Track history for reflection (use first agent's perspective for simplicity)
                episode_history.append({
                    "state": state.copy(),
                    "actions": actions,
                    "rewards": rewards,
                    "next_state": next_state.copy(),
                })

                # Track repeated actions (simplified - count if any agent repeats)
                if len(episode_history) > 1:
                    prev_actions = episode_history[-2]["actions"]
                    for i, action in enumerate(actions):
                        if action == prev_actions[i]:
                            repeated_action_count += 1

                # Track metrics (sum of all agents' rewards)
                episode_reward += sum(rewards)
                episode_length += 1
                episode_profit += sum(info.get("profits", [0.0]))

                state = next_state

                # 🧠 MEMORY STEP 2: Periodically reflect and update strategy
                if step > 0 and step % 5 == 0 and len(episode_history) >= 2:
                    # Generate insights from memory
                    insights = self.reflection.generate_insights(self.episodic_memory)

                    # Update all agents' strategies with insights
                    for controller_agent in self.controller_agents.values():
                        controller_agent.update_strategy(insights)

                if done:
                    break
            
            # Learn from experiences
            for agent_id, agent in self.agents.items():
                agent.replay(self.config.batch_size)
                agent.decay_epsilon()
            
            # 🧠 MEMORY STEP 3: End of episode reflection
            final_insights = self.reflection.generate_insights(self.episodic_memory)
            for agent_id in self.controller_agents:
                self.controller_agents[agent_id].update_strategy(final_insights)
            
            # Mark episode end in memory
            self.episodic_memory.end_episode()
            
            # Track episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_profits.append(episode_profit)
            self.episode_insights.append(len(final_insights))
            self.episode_repeated_actions.append(repeated_action_count)
            
            # Update best reward
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self._save_checkpoint(episode)
            
            # Logging with memory insights
            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config.log_interval :])
                avg_profit = np.mean(self.episode_profits[-self.config.log_interval :])
                avg_insights = np.mean(self.episode_insights[-self.config.log_interval :])
                avg_repeated = np.mean(self.episode_repeated_actions[-self.config.log_interval :])
                sample_insights = "; ".join(final_insights[:3]) if final_insights else "None"

                print(
                    f"Episode {episode + 1}/{self.config.num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Profit: ${avg_profit:,.0f} | "
                    f"Insights: {avg_insights:.1f} | "
                    f"Repeated: {avg_repeated:.1f} | "
                    f"Sample insights: {sample_insights}"
                )

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate trained agents without exploration.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation results
        """
        print(f"\nEvaluating for {num_episodes} episodes (no exploration)...\n")
        
        eval_rewards = []
        eval_profits = []
        
        for episode in range(num_episodes):
            episode_reward = 0.0
            episode_profit = 0.0

            state = self.env.reset()

            for step in range(self.config.max_steps):
                # All agents act simultaneously (greedy actions, no exploration)
                actions = []
                for agent in self.agents.values():
                    action = agent.act(state, training=False)
                    actions.append(action)

                next_state, rewards, done, info = self.env.step(actions)
                episode_reward += sum(rewards)
                episode_profit += sum(info.get("profits", [0.0]))

                state = next_state

                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_profits.append(episode_profit)
        
        avg_eval_reward = np.mean(eval_rewards)
        avg_eval_profit = np.mean(eval_profits)
        
        print(
            f"Evaluation Results:\n"
            f"  Average Reward: {avg_eval_reward:.2f}\n"
            f"  Average Profit: ${avg_eval_profit:,.0f}\n"
        )
        
        return {
            "eval_rewards": eval_rewards,
            "eval_profits": eval_profits,
            "avg_reward": float(avg_eval_reward),
            "avg_profit": float(avg_eval_profit),
        }
    
    def _save_checkpoint(self, episode: int) -> None:
        """
        Save checkpoint with best reward.
        
        Args:
            episode: Current episode number
        """
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            filepath = checkpoint_dir / f"{agent_id}_episode_{episode}.json"
            agent.save(str(filepath))
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile training results into dictionary."""
        return {
            "num_episodes": self.config.num_episodes,
            "num_agents": self.config.num_agents,
            "total_episodes": len(self.episode_rewards),
            "avg_reward": float(np.mean(self.episode_rewards)),
            "max_reward": float(np.max(self.episode_rewards)),
            "avg_length": float(np.mean(self.episode_lengths)),
            "avg_profit": float(np.mean(self.episode_profits)),
            "avg_insights_generated": float(np.mean(self.episode_insights)),
            "avg_repeated_actions": float(np.mean(self.episode_repeated_actions)),
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_profits": self.episode_profits,
            "episode_insights": self.episode_insights,
            "episode_repeated_actions": self.episode_repeated_actions,
            "memory_stats": self.episodic_memory.get_statistics(),
        }
    
    def save_results(self) -> None:
        """Save training results to file."""
        results_file = self.output_dir / "training_results.json"
        results = self._compile_results()
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
    
    def save_agents(self) -> None:
        """Save all trained agents."""
        models_dir = self.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            filepath = models_dir / f"{agent_id}_final.json"
            agent.save(str(filepath))

    def plot_rewards(self) -> None:
        """
        Plot training rewards per episode and save as PNG.

        Creates visualization with:
        - Episode rewards line plot
        - Moving average overlay
        - Clear axis labels and title
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available, skipping reward plot")
            return

        if not self.episode_rewards:
            print("No episode rewards to plot")
            return

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot episode rewards
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        ax.plot(
            episodes,
            self.episode_rewards,
            label="Episode Reward",
            color="steelblue",
            alpha=0.7,
            linewidth=1.5,
        )

        # Plot moving average (window=10)
        if len(self.episode_rewards) >= 10:
            moving_avg = np.convolve(
                self.episode_rewards, np.ones(10) / 10, mode="valid"
            )
            ax.plot(
                episodes[9:],
                moving_avg,
                label="10-Episode Moving Average",
                color="coral",
                linewidth=2.5,
            )

        # Labels and title
        ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
        ax.set_ylabel("Total Reward", fontsize=12, fontweight="bold")
        ax.set_title("Training Rewards Over Episodes", fontsize=14, fontweight="bold")
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
        plot_file = self.output_dir / "reward_plot.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        print(f"\nReward plot saved to {plot_file}")
        plt.close()


def main():
    """Main training entry point."""
    # Load configuration
    config = TrainingConfig()
    
    # Create environment
    env = StartupEnv(num_startups=config.num_agents, max_steps=config.max_steps)
    
    # Create trainer
    trainer = Trainer(config, env)
    
    # Run training
    trainer.train()
    
    # Evaluate
    trainer.evaluate(num_episodes=10)
    
    # Save results and models
    trainer.save_results()
    trainer.save_agents()
    
    # Plot rewards
    trainer.plot_rewards()
    
    print(f"\nTraining artifacts saved to {trainer.output_dir}")


if __name__ == "__main__":
    main()
