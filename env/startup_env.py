"""
OpenAI Gym-style environment for startup simulation.

State Variables:
- market_demand: Customer demand (0-100)
- cash: Available capital (0-1M)
- product_quality: Quality rating (0-100)
- competition: Market competition level (0-100)
- units_sold: Units sold this step

Actions:
- 0: build_product (spend cash, increase quality, increase units)
- 1: improve_quality (spend cash, increase quality)
- 2: run_marketing (spend cash, increase market demand)
- 3: reduce_price (lower price, increase demand, lower margin)
- 4: analyze_market (no cost, gain market insight)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces  # type: ignore
from typing import Dict, Tuple, Any


class StartupEnv(gym.Env):
    """
    Simple startup simulation environment.
    
    The agent controls a startup and must balance cash, product quality,
    and market presence to maximize profit and growth.
    """
    
    metadata = {"render_modes": ["human"]}
    
    # Action definitions
    ACTIONS = {
        0: "build_product",
        1: "improve_quality",
        2: "run_marketing",
        3: "reduce_price",
        4: "analyze_market",
    }
    
    def __init__(self, max_steps: int = 50, seed: int = None):
        """
        Initialize the startup environment.
        
        Args:
            max_steps: Maximum number of steps per episode
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Observation space: 6 continuous values
        # [market_demand, cash, product_quality, competition, units_sold, price]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0.1]),
            high=np.array([100, 1000000, 100, 100, 10000, 100]),
            dtype=np.float32,
        )
        
        # State variables
        self.state: Dict[str, float] = {}
        self._init_state()
    
    def _init_state(self) -> None:
        """Initialize state to starting conditions."""
        self.state = {
            "market_demand": 20.0,        # Initial market interest
            "cash": 100000.0,              # Starting capital
            "product_quality": 30.0,       # Initial product quality (0-100)
            "competition": 40.0,           # Market competition level
            "units_sold": 0.0,             # Units sold this step
            "price": 50.0,                 # Unit price
            "accumulated_revenue": 0.0,    # Total revenue
            "accumulated_costs": 0.0,      # Total costs
        }
    
    def reset(self) -> Dict[str, float]:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state dictionary
        """
        self.current_step = 0
        self._init_state()
        return self._get_obs()
    
    def _get_obs(self) -> Dict[str, float]:
        """Get current observation as dictionary."""
        return {
            "market_demand": float(self.state["market_demand"]),
            "cash": float(self.state["cash"]),
            "product_quality": float(self.state["product_quality"]),
            "competition": float(self.state["competition"]),
            "units_sold": float(self.state["units_sold"]),
            "price": float(self.state["price"]),
        }
    
    def step(self, action: int) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        Args:
            action: Action to execute (0-4)
        
        Returns:
            observation, reward, done, info
        """
        self.current_step += 1
        
        # Execute action
        cost = self._execute_action(action)
        
        # Update market dynamics
        self._update_market()
        
        # Calculate sales and revenue
        self._calculate_sales()
        
        # Calculate reward
        reward = self._calculate_reward(cost)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or self.state["cash"] < 0
        
        # Prepare info dict
        info = {
            "action": self.ACTIONS[action],
            "step": self.current_step,
            "revenue": self.state["units_sold"] * self.state["price"],
            "profit": (self.state["units_sold"] * self.state["price"]) - cost,
            "total_cash": self.state["cash"],
        }
        
        return self._get_obs(), reward, done, info
    
    def _execute_action(self, action: int) -> float:
        """
        Execute the chosen action and return cost.
        
        Args:
            action: Action index (0-4)
        
        Returns:
            Cost of the action
        """
        cost = 0.0
        
        if action == 0:  # build_product
            cost = 15000.0
            self.state["product_quality"] = min(
                100, self.state["product_quality"] + np.random.uniform(5, 15)
            )
            self.state["units_sold"] *= 1.2  # Boost sales
        
        elif action == 1:  # improve_quality
            cost = 8000.0
            self.state["product_quality"] = min(
                100, self.state["product_quality"] + np.random.uniform(3, 10)
            )
        
        elif action == 2:  # run_marketing
            cost = 10000.0
            boost = np.random.uniform(5, 15)
            self.state["market_demand"] = min(100, self.state["market_demand"] + boost)
        
        elif action == 3:  # reduce_price
            cost = 0.0
            self.state["price"] = max(10.0, self.state["price"] * 0.85)
        
        elif action == 4:  # analyze_market
            cost = 2000.0
            # Just a small insight - doesn't directly affect state
            pass
        
        # Deduct cost from cash
        self.state["cash"] -= cost
        self.state["accumulated_costs"] += cost
        
        return cost
    
    def _update_market(self) -> None:
        """Update market dynamics over time."""
        # Market demand naturally decays without maintenance
        self.state["market_demand"] *= 0.98
        self.state["market_demand"] = max(0, self.state["market_demand"])
        
        # Competition increases slightly over time
        self.state["competition"] = min(
            100, self.state["competition"] + np.random.uniform(-2, 3)
        )
        
        # Quality degrades without maintenance
        self.state["product_quality"] *= 0.99
        self.state["product_quality"] = max(0, self.state["product_quality"])
    
    def _calculate_sales(self) -> None:
        """Calculate units sold based on current state."""
        # Base demand affected by market demand, quality, and price
        quality_factor = self.state["product_quality"] / 100.0
        demand_factor = self.state["market_demand"] / 100.0
        
        # Price elasticity: lower price = more sales
        price_factor = 100.0 / (self.state["price"] + 1)
        
        # Competition reduces demand
        competition_factor = 1.0 - (self.state["competition"] / 200.0)
        
        # Calculate units sold
        base_units = 50.0
        units = (
            base_units
            * quality_factor
            * demand_factor
            * (price_factor / 10.0)
            * competition_factor
        )
        
        # Add some randomness
        units = max(0, units + np.random.normal(0, units * 0.1))
        
        self.state["units_sold"] = units
        
        # Update cash with revenue
        revenue = units * self.state["price"]
        self.state["cash"] += revenue
        self.state["accumulated_revenue"] += revenue
    
    def _calculate_reward(self, cost: float) -> float:
        """
        Calculate reward based on profit and growth.
        
        Args:
            cost: Cost of this step's action
        
        Returns:
            Reward value
        """
        revenue = self.state["units_sold"] * self.state["price"]
        profit = revenue - cost
        
        # Growth reward: increase in units sold
        growth_reward = self.state["units_sold"] * 0.5
        
        # Profit reward
        profit_reward = max(0, profit / 100.0)
        
        # Cash preservation reward
        cash_reward = self.state["cash"] / 100000.0 * 10.0
        
        # Penalty for action cost
        cost_penalty = cost / 50000.0
        
        # Quality reward
        quality_reward = self.state["product_quality"] / 100.0 * 2.0
        
        total_reward = (
            growth_reward + profit_reward + cash_reward - cost_penalty + quality_reward
        )
        
        return float(total_reward)
    
    def render(self, mode: str = "human") -> None:
        """Render the environment state."""
        print(
            f"\nStep {self.current_step}/{self.max_steps}\n"
            f"  Market Demand: {self.state['market_demand']:.1f}\n"
            f"  Cash: ${self.state['cash']:,.0f}\n"
            f"  Product Quality: {self.state['product_quality']:.1f}\n"
            f"  Competition: {self.state['competition']:.1f}\n"
            f"  Units Sold: {self.state['units_sold']:.0f}\n"
            f"  Price: ${self.state['price']:.2f}\n"
        )


if __name__ == "__main__":
    # Quick test
    env = StartupEnv(max_steps=10)
    obs = env.reset()
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"  Reward: {reward:.2f}, Total Cash: ${info['total_cash']:,.0f}")
        if done:
            break
    
    print("\nEpisode finished!")
