"""
Reward function for startup simulation environment.

Combines multiple reward signals:
1. Business rewards (cash, quality growth)
2. Learning rewards (exploration, avoiding repeated failures)
3. Adaptation rewards (recovery from crisis)
4. Penalties (action repetition)
"""

from typing import Dict, List, Any
from collections import Counter


class RewardFunction:
    """
    Multi-component reward function for startup simulation.
    
    Encourages agents to:
    - Grow cash and product quality
    - Explore new action sequences
    - Recover from low-cash situations
    - Avoid repetitive, ineffective strategies
    """
    
    def __init__(
        self,
        business_weight: float = 1.0,
        learning_weight: float = 0.5,
        adaptation_weight: float = 0.3,
        repetition_penalty_weight: float = -0.5,
    ):
        """
        Initialize reward function with component weights.
        
        Args:
            business_weight: Weight for business metrics rewards
            learning_weight: Weight for exploration/learning rewards
            adaptation_weight: Weight for crisis recovery rewards
            repetition_penalty_weight: Weight for action repetition penalties
        """
        self.business_weight = business_weight
        self.learning_weight = learning_weight
        self.adaptation_weight = adaptation_weight
        self.repetition_penalty_weight = repetition_penalty_weight
        
        # Track failed action sequences for learning reward
        self.failed_actions: Dict[int, int] = {}  # action -> failure count
    
    def calculate(
        self,
        previous_state: Dict[str, float],
        current_state: Dict[str, float],
        action: int,
        history: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate composite reward based on multiple signals.
        
        Args:
            previous_state: State before action execution
            current_state: State after action execution
            action: Action taken (0-4)
            history: List of (state, action, reward) tuples from episode
        
        Returns:
            Composite reward value
        """
        business_reward = self._business_reward(previous_state, current_state)
        learning_reward = self._learning_reward(action, history)
        adaptation_reward = self._adaptation_reward(previous_state, current_state)
        repetition_penalty = self._repetition_penalty(action, history)
        failed_action_penalty = self._failed_action_penalty(action, history)
        
        # Combine rewards with weights
        total_reward = (
            self.business_weight * business_reward
            + self.learning_weight * learning_reward
            + self.adaptation_weight * adaptation_reward
            + self.repetition_penalty_weight * repetition_penalty
            + self.repetition_penalty_weight * failed_action_penalty  # Use same weight as repetition
        )
        
        return float(total_reward)
    
    def _business_reward(
        self,
        previous_state: Dict[str, float],
        current_state: Dict[str, float],
    ) -> float:
        """
        Reward based on business metrics improvement.
        
        Components:
        - Cash increase (normalized)
        - Product quality increase
        - Units sold (growth signal)
        
        Args:
            previous_state: State before action
            current_state: State after action
        
        Returns:
            Business reward value
        """
        # Cash growth reward
        cash_delta = current_state["cash"] - previous_state["cash"]
        cash_reward = cash_delta / 50000.0  # Normalize to reasonable scale
        
        # Quality improvement reward
        quality_delta = current_state["product_quality"] - previous_state["product_quality"]
        quality_reward = quality_delta * 0.05  # Scale down quality changes
        
        # Units sold growth reward
        units_reward = current_state["units_sold"] * 0.01
        
        business_reward = cash_reward + quality_reward + units_reward
        
        return float(business_reward)
    
    def _learning_reward(
        self,
        action: int,
        history: List[Dict[str, Any]],
    ) -> float:
        """
        Reward for exploring actions different from repeated failures.
        
        Encourages the agent to:
        - Try new action sequences
        - Avoid repeating actions that recently failed
        - Maintain action diversity
        
        Args:
            action: Action taken
            history: Episode history
        
        Returns:
            Learning reward (0 or positive bonus)
        """
        # If no history, give small exploration bonus
        if len(history) < 2:
            return 0.5
        
        # Check recent actions (last 5 steps)
        recent_actions = [h.get("action", -1) for h in history[-5:]]
        
        # Count how many times current action appears in recent history
        action_count = recent_actions.count(action)
        
        # Reward for action variety
        if action_count == 0:
            # Brand new action in recent history - good exploration
            return 1.0
        elif action_count == 1:
            # Action used once recently - acceptable
            return 0.2
        else:
            # Action used multiple times - no bonus
            return 0.0
    
    def _adaptation_reward(
        self,
        previous_state: Dict[str, float],
        current_state: Dict[str, float],
    ) -> float:
        """
        Reward for recovering from crisis situations.
        
        Particularly rewards:
        - Increasing cash when it was critically low
        - Maintaining operations during low-cash periods
        
        Args:
            previous_state: State before action
            current_state: State after action
        
        Returns:
            Adaptation reward (0 or positive bonus)
        """
        previous_cash = previous_state["cash"]
        current_cash = current_state["cash"]
        cash_threshold = 20000.0  # Crisis threshold
        
        # If previously in crisis and recovered
        if previous_cash < cash_threshold and current_cash > previous_cash:
            crisis_recovery_bonus = 2.0
            return crisis_recovery_bonus
        
        # If in crisis and maintained (didn't get worse)
        if previous_cash < cash_threshold and current_cash >= previous_cash:
            crisis_maintenance_bonus = 0.5
            return crisis_maintenance_bonus
        
        # If avoiding crisis by building cash buffer
        if previous_cash > cash_threshold and current_cash > previous_cash:
            buffer_building_bonus = (current_cash - previous_cash) / 100000.0 * 0.5
            return buffer_building_bonus
        
        return 0.0
    
    def _repetition_penalty(
        self,
        action: int,
        history: List[Dict[str, Any]],
    ) -> float:
        """
        Penalize repeating the same action too many times consecutively.
        
        Discourages:
        - Taking the same action 3+ times in a row
        - Stuck in action loops
        
        Args:
            action: Action taken
            history: Episode history
        
        Returns:
            Penalty value (0 or positive penalty)
        """
        if len(history) < 2:
            return 0.0
        
        # Count consecutive repetitions of current action
        consecutive_count = 1
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("action", -1) == action:
                consecutive_count += 1
            else:
                break
        
        # Penalty increases with consecutive repetitions
        if consecutive_count >= 3:
            # Heavy penalty for 3+ repetitions
            penalty = (consecutive_count - 2) * 2.0
            return penalty
        
        return 0.0
    
    def _failed_action_penalty(
        self,
        action: int,
        history: List[Dict[str, Any]],
    ) -> float:
        """
        Penalize repeating same action if last 3 outcomes were negative.
        
        Encourages trying different action after repeated failures.
        
        Args:
            action: Action taken
            history: Episode history
        
        Returns:
            Penalty value (0 or positive penalty)
        """
        if len(history) < 3:
            return 0.0
        
        # Check if this action was used in last 3 steps
        recent_actions = [h.get("action", -1) for h in history[-3:]]
        recent_rewards = [h.get("reward", 0.0) for h in history[-3:]]
        
        # If action is in recent history and all recent rewards are negative
        if action in recent_actions and all(r < 0 for r in recent_rewards):
            # Strong penalty for repeating failed action
            return 3.0
        
        return 0.0
    
    def record_failed_action(self, action: int) -> None:
        """
        Record that an action failed to improve the state.
        
        Used to track which actions tend to fail, informing learning reward.
        
        Args:
            action: Action that failed
        """
        if action not in self.failed_actions:
            self.failed_actions[action] = 0
        self.failed_actions[action] += 1
    
    def reset_failed_actions(self) -> None:
        """Reset the failed action history."""
        self.failed_actions = {}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of reward function configuration.
        
        Returns:
            Dictionary with weights and statistics
        """
        return {
            "business_weight": self.business_weight,
            "learning_weight": self.learning_weight,
            "adaptation_weight": self.adaptation_weight,
            "repetition_penalty_weight": self.repetition_penalty_weight,
            "failed_actions": dict(self.failed_actions),
        }


def create_reward_function(**kwargs) -> RewardFunction:
    """
    Factory function to create reward function with custom weights.
    
    Args:
        **kwargs: Keyword arguments for RewardFunction.__init__
    
    Returns:
        Configured RewardFunction instance
    """
    return RewardFunction(**kwargs)


if __name__ == "__main__":
    # Test reward function
    reward_fn = RewardFunction()
    
    # Test case 1: Positive business result
    prev_state = {
        "cash": 100000,
        "product_quality": 50,
        "units_sold": 100,
        "market_demand": 50,
        "competition": 40,
        "price": 50,
    }
    
    curr_state = {
        "cash": 120000,
        "product_quality": 60,
        "units_sold": 150,
        "market_demand": 55,
        "competition": 42,
        "price": 45,
    }
    
    history = [
        {"action": 0, "reward": 1.0},
        {"action": 1, "reward": 0.5},
    ]
    
    reward = reward_fn.calculate(prev_state, curr_state, action=2, history=history)
    print(f"Test 1 - Good business result: Reward = {reward:.2f}")
    
    # Test case 2: Repetition penalty
    prev_state_rep = {
        "cash": 90000,
        "product_quality": 45,
        "units_sold": 80,
        "market_demand": 40,
        "competition": 45,
        "price": 50,
    }
    
    curr_state_rep = {
        "cash": 85000,
        "product_quality": 45,
        "units_sold": 80,
        "market_demand": 40,
        "competition": 45,
        "price": 50,
    }
    
    history_rep = [
        {"action": 0, "reward": -0.5},
        {"action": 0, "reward": -0.3},
        {"action": 0, "reward": -0.2},
    ]
    
    reward_rep = reward_fn.calculate(prev_state_rep, curr_state_rep, action=0, history=history_rep)
    print(f"Test 2 - Repetition penalty: Reward = {reward_rep:.2f}")
    
    # Test case 3: Crisis recovery
    prev_state_crisis = {
        "cash": 15000,
        "product_quality": 30,
        "units_sold": 50,
        "market_demand": 30,
        "competition": 50,
        "price": 40,
    }
    
    curr_state_crisis = {
        "cash": 35000,
        "product_quality": 35,
        "units_sold": 100,
        "market_demand": 35,
        "competition": 48,
        "price": 45,
    }
    
    history_crisis = [{"action": 2, "reward": 0.5}]
    
    reward_crisis = reward_fn.calculate(
        prev_state_crisis, curr_state_crisis, action=2, history=history_crisis
    )
    print(f"Test 3 - Crisis recovery: Reward = {reward_crisis:.2f}")
