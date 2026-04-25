"""
Training configuration for startup simulation agents.

Centralizes hyperparameters and training settings.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration for DQN training.
    
    Attributes:
        num_episodes: Total episodes to train
        num_agents: Number of agents to train
        max_steps: Maximum steps per episode
        batch_size: Batch size for experience replay
        learning_rate: Q-learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Exploration decay per episode
        replay_buffer_size: Maximum replay buffer size
        log_interval: Episodes between logging
        output_dir: Directory for training outputs
    """
    
    # Training parameters
    num_episodes: int = 1000
    num_agents: int = 1
    max_steps: int = 50
    
    # Learning parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.99
    
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    
    # Replay buffer
    replay_buffer_size: int = 10000
    
    # Logging
    log_interval: int = 50
    
    # Output
    output_dir: str = "training_output"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.num_episodes > 0, "num_episodes must be positive"
        assert self.num_agents > 0, "num_agents must be positive"
        assert self.max_steps > 0, "max_steps must be positive"
        assert 0 < self.learning_rate < 1, "learning_rate must be between 0 and 1"
        assert 0 < self.gamma < 1, "gamma must be between 0 and 1"
        assert self.epsilon_min < self.epsilon_start, "epsilon_min must be < epsilon_start"
        assert 0 < self.epsilon_decay < 1, "epsilon_decay must be between 0 and 1"
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "num_episodes": self.num_episodes,
            "num_agents": self.num_agents,
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "replay_buffer_size": self.replay_buffer_size,
            "log_interval": self.log_interval,
            "output_dir": self.output_dir,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


# Preset configurations for different scenarios
class PresetConfigs:
    """Preset training configurations."""
    
    @staticmethod
    def quick_test() -> TrainingConfig:
        """Quick testing configuration."""
        return TrainingConfig(
            num_episodes=100,
            num_agents=1,
            max_steps=50,
            batch_size=16,
            log_interval=10,
        )
    
    @staticmethod
    def single_agent() -> TrainingConfig:
        """Single agent training."""
        return TrainingConfig(
            num_episodes=1000,
            num_agents=1,
            max_steps=50,
            batch_size=32,
            learning_rate=0.001,
            log_interval=50,
        )
    
    @staticmethod
    def multi_agent() -> TrainingConfig:
        """Multi-agent training."""
        return TrainingConfig(
            num_episodes=2000,
            num_agents=3,
            max_steps=50,
            batch_size=64,
            learning_rate=0.0005,
            log_interval=100,
        )
    
    @staticmethod
    def production() -> TrainingConfig:
        """Production training."""
        return TrainingConfig(
            num_episodes=5000,
            num_agents=5,
            max_steps=50,
            batch_size=128,
            learning_rate=0.0001,
            epsilon_decay=0.9995,
            replay_buffer_size=50000,
            log_interval=100,
        )


if __name__ == "__main__":
    # Test configurations
    print("Default Config:")
    default_cfg = TrainingConfig()
    print(default_cfg)
    
    print("\nQuick Test Config:")
    quick_cfg = PresetConfigs.quick_test()
    print(quick_cfg)
    
    print("\nMulti-Agent Config:")
    multi_cfg = PresetConfigs.multi_agent()
    print(multi_cfg)
