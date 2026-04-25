"""
Episodic Memory Store for startup simulation agents.

Stores specific past experiences and events.
Allows retrieval of recent experiences and similar past states.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class Experience:
    """Single experience tuple."""
    
    state: Dict[str, float]
    action: int
    reward: float
    next_state: Dict[str, float]
    timestamp: float = None
    episode: int = None
    step: int = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()


class EpisodicMemory:
    """
    Episodic memory store for experiences.
    
    Stores and retrieves specific past events from the agent's experience.
    Simple list-based implementation (future: vector DB).
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize episodic memory.
        
        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self.experiences: List[Experience] = []
        self.current_episode = 0
        self.current_step = 0
    
    def add_experience(
        self,
        state: Dict[str, float],
        action: int,
        reward: float,
        next_state: Dict[str, float],
    ) -> None:
        """
        Add experience to memory.
        
        Args:
            state: State before action
            action: Action taken (0-4)
            reward: Reward received
            next_state: Resulting state
        """
        experience = Experience(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            episode=self.current_episode,
            step=self.current_step,
        )
        
        self.experiences.append(experience)
        self.current_step += 1
        
        # Maintain max size (FIFO eviction)
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)
    
    def get_recent(self, n: int = 5) -> List[Experience]:
        """
        Get N most recent experiences.
        
        Args:
            n: Number of recent experiences to retrieve
        
        Returns:
            List of most recent experiences
        """
        if not self.experiences:
            return []
        
        return self.experiences[-n:]
    
    def get_similar(
        self,
        state: Dict[str, float],
        n: int = 5,
        threshold: float = None,
    ) -> List[Tuple[Experience, float]]:
        """
        Get experiences with similar states.
        
        Uses simple Euclidean distance in state space.
        
        Args:
            state: Query state
            n: Number of similar experiences to retrieve
            threshold: Maximum distance threshold (None = no threshold)
        
        Returns:
            List of (experience, similarity_score) tuples, sorted by similarity
        """
        if not self.experiences:
            return []
        
        # Calculate similarity to each experience
        similarities = []
        for exp in self.experiences:
            distance = self._state_distance(state, exp.state)
            similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
            
            if threshold is None or similarity >= threshold:
                similarities.append((exp, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n]
    
    def get_by_action(self, action: int, n: int = None) -> List[Experience]:
        """
        Get all experiences for a specific action.
        
        Args:
            action: Action index (0-4)
            n: Limit to N most recent (None = all)
        
        Returns:
            List of experiences for that action
        """
        action_experiences = [exp for exp in self.experiences if exp.action == action]
        
        if n is not None:
            return action_experiences[-n:]
        return action_experiences
    
    def get_high_reward_experiences(
        self,
        percentile: float = 75.0,
        n: int = None,
    ) -> List[Experience]:
        """
        Get high-reward experiences above percentile.
        
        Args:
            percentile: Reward percentile threshold (0-100)
            n: Limit to N most recent (None = all)
        
        Returns:
            List of high-reward experiences
        """
        if not self.experiences:
            return []
        
        rewards = [exp.reward for exp in self.experiences]
        threshold = np.percentile(rewards, percentile)
        
        high_reward_exp = [exp for exp in self.experiences if exp.reward >= threshold]
        
        if n is not None:
            return high_reward_exp[-n:]
        return high_reward_exp
    
    def get_episode_experiences(self, episode: int) -> List[Experience]:
        """
        Get all experiences from a specific episode.
        
        Args:
            episode: Episode number
        
        Returns:
            List of experiences from that episode
        """
        return [exp for exp in self.experiences if exp.episode == episode]
    
    def end_episode(self) -> None:
        """Mark end of episode and increment counter."""
        self.current_episode += 1
        self.current_step = 0
    
    def clear(self) -> None:
        """Clear all experiences."""
        self.experiences = []
        self.current_episode = 0
        self.current_step = 0
    
    def _state_distance(
        self,
        state1: Dict[str, float],
        state2: Dict[str, float],
    ) -> float:
        """
        Calculate Euclidean distance between two states.
        
        Uses standard state variables in consistent order.
        
        Args:
            state1: First state
            state2: Second state
        
        Returns:
            Euclidean distance
        """
        # Standard state features in order
        features = [
            "market_demand",
            "cash",
            "product_quality",
            "competition",
            "units_sold",
            "price",
        ]
        
        distance = 0.0
        for feature in features:
            v1 = state1.get(feature, 0.0)
            v2 = state2.get(feature, 0.0)
            distance += (v1 - v2) ** 2
        
        return np.sqrt(distance)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored experiences.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.experiences:
            return {
                "total_experiences": 0,
                "memory_full": False,
                "avg_reward": 0.0,
                "max_reward": 0.0,
                "min_reward": 0.0,
            }
        
        rewards = [exp.reward for exp in self.experiences]
        actions = [exp.action for exp in self.experiences]
        
        return {
            "total_experiences": len(self.experiences),
            "memory_full": len(self.experiences) >= self.max_size,
            "avg_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "std_reward": float(np.std(rewards)),
            "action_distribution": self._get_action_distribution(actions),
            "current_episode": self.current_episode,
        }
    
    def _get_action_distribution(self, actions: List[int]) -> Dict[int, int]:
        """Get count of each action."""
        from collections import Counter
        counts = Counter(actions)
        return {i: counts.get(i, 0) for i in range(5)}
    
    def recall_narrative(self, query_state: Dict[str, float]) -> str:
        """
        Generate a narrative description of similar past experiences.
        
        Args:
            query_state: State to find similar experiences for
        
        Returns:
            Narrative string
        """
        similar = self.get_similar(query_state, n=3)
        
        if not similar:
            return "No similar past experiences found."
        
        action_names = {
            0: "build_product",
            1: "improve_quality",
            2: "run_marketing",
            3: "reduce_price",
            4: "analyze_market",
        }
        
        narratives = ["Similar past experiences:"]
        for i, (exp, similarity) in enumerate(similar, 1):
            narrative = (
                f"  {i}. {action_names[exp.action]} "
                f"(similarity: {similarity:.2f}, "
                f"reward: {exp.reward:.2f})"
            )
            narratives.append(narrative)
        
        return "\n".join(narratives)


def demo_episodic_memory():
    """Demo the episodic memory."""
    print("=" * 70)
    print("EPISODIC MEMORY DEMO")
    print("=" * 70)
    
    memory = EpisodicMemory(max_size=1000)
    
    # Simulate some experiences
    states = [
        {
            "market_demand": 40.0,
            "cash": 100000,
            "product_quality": 50.0,
            "competition": 45.0,
            "units_sold": 100.0,
            "price": 50.0,
        },
        {
            "market_demand": 45.0,
            "cash": 95000,
            "product_quality": 55.0,
            "competition": 44.0,
            "units_sold": 110.0,
            "price": 50.0,
        },
        {
            "market_demand": 50.0,
            "cash": 90000,
            "product_quality": 60.0,
            "competition": 43.0,
            "units_sold": 120.0,
            "price": 50.0,
        },
        {
            "market_demand": 30.0,  # Different scenario
            "cash": 150000,
            "product_quality": 70.0,
            "competition": 50.0,
            "units_sold": 200.0,
            "price": 60.0,
        },
    ]
    
    # Add experiences
    print("\nAdding experiences...")
    for i, state in enumerate(states):
        next_state = states[(i + 1) % len(states)]
        reward = 1.5 + i * 0.5
        memory.add_experience(state, action=i % 5, reward=reward, next_state=next_state)
    
    print(f"Added {len(memory.experiences)} experiences")
    
    # Get recent experiences
    print("\nRecent experiences (last 2):")
    recent = memory.get_recent(2)
    for exp in recent:
        print(f"  Action: {exp.action}, Reward: {exp.reward:.2f}")
    
    # Get similar experiences
    print("\nSimilar experiences to first state:")
    query_state = states[0]
    similar = memory.get_similar(query_state, n=2)
    for exp, similarity in similar:
        print(f"  Action: {exp.action}, Similarity: {similarity:.3f}, Reward: {exp.reward:.2f}")
    
    # Recall narrative
    print("\nNarrative recall:")
    narrative = memory.recall_narrative(query_state)
    print(narrative)
    
    # Statistics
    print("\nMemory statistics:")
    stats = memory.get_statistics()
    for key, value in stats.items():
        if key != "action_distribution":
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_episodic_memory()
