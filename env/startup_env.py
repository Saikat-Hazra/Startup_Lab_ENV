"""
OpenAI Gym-style environment for startup simulation.

State Variables (per startup):
- cash: Available capital (0-1M)
- product_quality: Quality rating (0-100)
- units_sold: Units sold this step

Shared State Variables:
- market_demand: Customer demand (0-100)
- competition: Market competition level (0-100)

Actions (per startup):
- 0: build_product (spend cash, increase quality, increase units)
- 1: improve_quality (spend cash, increase quality)
- 2: run_marketing (spend cash, increase market demand)
- 3: reduce_price (lower price, increase demand, lower margin)
- 4: analyze_market (no cost, gain market insight)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces  # type: ignore
from typing import Dict, List, Tuple, Any


class StartupEnv(gym.Env):
    """
    Multi-startup simulation environment.

    Multiple startups compete in the same market, each making decisions
    to maximize their own profit and growth.
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

    def __init__(self, num_startups: int = 2, max_steps: int = 50, seed: int = None):
        """
        Initialize the multi-startup environment.

        Args:
            num_startups: Number of competing startups
            max_steps: Maximum number of steps per episode
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.num_startups = num_startups
        self.max_steps = max_steps
        self.current_step = 0

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Action space: 5 discrete actions per startup
        self.action_space = spaces.MultiDiscrete([5] * num_startups)

        # Observation space: per-startup states + shared market state
        # Each startup: [cash, product_quality, units_sold, price]
        # Shared: [market_demand, competition]
        startup_obs_low = [0, 0, 0, 0.1]  # cash, quality, units, price
        startup_obs_high = [1000000, 100, 10000, 100]
        shared_obs_low = [0, 0]  # market_demand, competition
        shared_obs_high = [100, 100]

        # Total observation space
        obs_low = startup_obs_low * num_startups + shared_obs_low
        obs_high = startup_obs_high * num_startups + shared_obs_high

        self.observation_space = spaces.Box(
            low=np.array(obs_low),
            high=np.array(obs_high),
            dtype=np.float32,
        )

        # State variables
        self.startup_states: List[Dict[str, float]] = []
        self.shared_state: Dict[str, float] = {}
        self._init_state()
    
    def _init_state(self) -> None:
        """Initialize state to starting conditions for all startups."""
        # Initialize shared market state
        self.shared_state = {
            "market_demand": 20.0,        # Initial market interest
            "competition": 40.0,          # Market competition level
        }

        # Initialize individual startup states
        self.startup_states = []
        for i in range(self.num_startups):
            startup_state = {
                "cash": 100000.0,              # Starting capital
                "product_quality": 30.0,       # Initial product quality (0-100)
                "units_sold": 0.0,             # Units sold this step
                "price": 50.0,                 # Unit price
                "accumulated_revenue": 0.0,    # Total revenue
                "accumulated_costs": 0.0,      # Total costs
            }
            self.startup_states.append(startup_state)

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation array
        """
        self.current_step = 0
        self._init_state()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Get current observation as flattened array."""
        obs = []

        # Add each startup's state
        for startup in self.startup_states:
            obs.extend([
                float(startup["cash"]),
                float(startup["product_quality"]),
                float(startup["units_sold"]),
                float(startup["price"]),
            ])

        # Add shared market state
        obs.extend([
            float(self.shared_state["market_demand"]),
            float(self.shared_state["competition"]),
        ])

        return np.array(obs, dtype=np.float32)
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[float], bool, Dict[str, Any]]:
        """
        Execute one step of the environment for all startups.

        Args:
            actions: List of actions to execute (one per startup)

        Returns:
            observation, rewards, done, info
        """
        self.current_step += 1

        # Execute actions for all startups
        costs = []
        for i, action in enumerate(actions):
            cost = self._execute_action(i, action)
            costs.append(cost)

        # Update shared market dynamics
        self._update_market()

        # Calculate sales and competition for all startups
        self._calculate_sales_competition()

        # Calculate rewards for each startup
        rewards = []
        for i, cost in enumerate(costs):
            reward = self._calculate_reward(i, cost)
            rewards.append(reward)

        # Check if episode is done (any startup bankrupt or max steps reached)
        any_bankrupt = any(startup["cash"] < 0 for startup in self.startup_states)
        done = self.current_step >= self.max_steps or any_bankrupt

        # Prepare info dict
        info = {
            "step": self.current_step,
            "actions": [self.ACTIONS[action] for action in actions],
            "costs": costs,
            "rewards": rewards,
            "revenues": [startup["units_sold"] * startup["price"] for startup in self.startup_states],
            "profits": [(startup["units_sold"] * startup["price"]) - cost
                       for startup, cost in zip(self.startup_states, costs)],
            "total_cash": [startup["cash"] for startup in self.startup_states],
        }

        return self._get_obs(), rewards, done, info
    
    def _execute_action(self, startup_idx: int, action: int) -> float:
        """
        Execute the chosen action for a specific startup and return cost.

        Args:
            startup_idx: Index of the startup
            action: Action index (0-4)

        Returns:
            Cost of the action
        """
        startup = self.startup_states[startup_idx]
        cost = 0.0

        if action == 0:  # build_product
            cost = 15000.0
            startup["product_quality"] = min(
                100, startup["product_quality"] + np.random.uniform(5, 15)
            )
            startup["units_sold"] *= 1.2  # Boost sales

        elif action == 1:  # improve_quality
            cost = 8000.0
            startup["product_quality"] = min(
                100, startup["product_quality"] + np.random.uniform(3, 10)
            )

        elif action == 2:  # run_marketing
            cost = 10000.0
            boost = np.random.uniform(5, 15)
            self.shared_state["market_demand"] = min(100, self.shared_state["market_demand"] + boost)

        elif action == 3:  # reduce_price
            cost = 0.0
            startup["price"] = max(10.0, startup["price"] * 0.85)

        elif action == 4:  # analyze_market
            cost = 2000.0
            # Just a small insight - doesn't directly affect state
            pass

        # Deduct cost from startup's cash
        startup["cash"] -= cost
        startup["accumulated_costs"] += cost

        return cost
    
    def _update_market(self) -> None:
        """Update shared market dynamics over time."""
        # Market demand naturally decays without maintenance
        self.shared_state["market_demand"] *= 0.98
        self.shared_state["market_demand"] = max(0, self.shared_state["market_demand"])

        # Competition increases slightly over time
        self.shared_state["competition"] = min(
            100, self.shared_state["competition"] + np.random.uniform(-2, 3)
        )

        # Quality degrades for all startups without maintenance
        for startup in self.startup_states:
            startup["product_quality"] *= 0.99
            startup["product_quality"] = max(0, startup["product_quality"])

    def _calculate_sales_competition(self) -> None:
        """Calculate units sold for all startups with competition dynamics."""
        # Calculate base market potential
        demand_factor = self.shared_state["market_demand"] / 100.0
        competition_factor = 1.0 - (self.shared_state["competition"] / 200.0)
        base_market_size = 100.0 * demand_factor * competition_factor

        # Calculate attractiveness for each startup
        attractiveness = []
        for startup in self.startup_states:
            quality_factor = startup["product_quality"] / 100.0
            price_factor = 100.0 / (startup["price"] + 1)  # Price elasticity
            attr = quality_factor * (price_factor / 10.0)
            attractiveness.append(attr)

        # Normalize attractiveness to determine market share
        total_attr = sum(attractiveness)
        if total_attr > 0:
            market_shares = [attr / total_attr for attr in attractiveness]
        else:
            market_shares = [1.0 / self.num_startups] * self.num_startups

        # Allocate market to startups based on relative attractiveness
        for i, startup in enumerate(self.startup_states):
            # Base units from market share
            units = base_market_size * market_shares[i]

            # Add some randomness
            units = max(0, units + np.random.normal(0, units * 0.1))

            startup["units_sold"] = units

            # Update cash with revenue
            revenue = units * startup["price"]
            startup["cash"] += revenue
            startup["accumulated_revenue"] += revenue
    
    def _calculate_reward(self, startup_idx: int, cost: float) -> float:
        """
        Calculate reward for a specific startup based on profit and growth.

        Args:
            startup_idx: Index of the startup
            cost: Cost of this step's action

        Returns:
            Reward value
        """
        startup = self.startup_states[startup_idx]
        revenue = startup["units_sold"] * startup["price"]
        profit = revenue - cost

        # Growth reward: increase in units sold
        growth_reward = startup["units_sold"] * 0.5

        # Profit reward
        profit_reward = max(0, profit / 100.0)

        # Cash preservation reward
        cash_reward = startup["cash"] / 100000.0 * 10.0

        # Penalty for action cost
        cost_penalty = cost / 50000.0

        # Quality reward
        quality_reward = startup["product_quality"] / 100.0 * 2.0

        total_reward = (
            growth_reward + profit_reward + cash_reward - cost_penalty + quality_reward
        )

        return float(total_reward)
    
    def render(self, mode: str = "human") -> None:
        """Render the environment state for all startups."""
        print(f"\nStep {self.current_step}/{self.max_steps}")
        print(f"Market Demand: {self.shared_state['market_demand']:.1f}")
        print(f"Competition: {self.shared_state['competition']:.1f}")

        for i, startup in enumerate(self.startup_states):
            print(f"\nStartup {i+1}:")
            print(f"  Cash: ${startup['cash']:>12,.0f}")
            print(f"  Product Quality: {startup['product_quality']:.1f}")
            print(f"  Units Sold: {startup['units_sold']:.0f}")
            print(f"  Price: ${startup['price']:.2f}")


if __name__ == "__main__":
    # Quick test with 2 startups
    env = StartupEnv(num_startups=2, max_steps=10)
    obs = env.reset()

    print("Initial state:")
    env.render()

    for step in range(10):
        # Random actions for both startups
        actions = [env.action_space.sample()[i] for i in range(env.num_startups)]
        obs, rewards, done, info = env.step(actions)

        print(f"\nStep {step + 1}:")
        print(f"Actions: {[env.ACTIONS[a] for a in actions]}")
        print(f"Rewards: {[f'{r:.2f}' for r in rewards]}")
        env.render()

        if done:
            print(f"\nEpisode ended at step {step + 1}")
            break

    print("\nEpisode finished!")
