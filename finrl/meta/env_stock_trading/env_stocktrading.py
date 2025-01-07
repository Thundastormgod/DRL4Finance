import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List

class StockingTradingEnv(gym.Env):
    """A sophisticated stock trading environment combining features from multiple implementations"""
    
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        """Initialize the trading environment
        
        Args:
            df (pd.DataFrame): DataFrame containing the stock data
            stock_dim (int): Number of stocks
            hmax (int): Maximum number of shares to trade
            initial_amount (int): Initial cash amount
            num_stock_shares (list[int]): Initial number of shares for each stock
            buy_cost_pct (list[float]): Transaction cost for buying each stock
            sell_cost_pct (list[float]): Transaction cost for selling each stock
            reward_scaling (float): Scaling factor for rewards
            state_space (int): Dimension of state space
            action_space (int): Dimension of action space
            tech_indicator_list (list[str]): List of technical indicators to use
            turbulence_threshold (float, optional): Threshold for market turbulence
            risk_indicator_col (str, optional): Column name for risk indicator
            make_plots (bool, optional): Whether to generate plots
            print_verbosity (int, optional): How often to print updates
            day (int, optional): Starting day
            initial (bool, optional): Whether this is initial training
            previous_state (list, optional): Previous state for continuing training
            model_name (str, optional): Name of the model being used
            mode (str, optional): Training mode
            iteration (str, optional): Training iteration
        """
        # Basic parameters
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.num_stock_shares = num_stock_shares
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        
        # Risk management parameters
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        
        # Technical Analysis Parameters
        self.indicator_weights = {
            'trend': 0.35,
            'momentum': 0.25,
            'mean_reversion': 0.25,
            'volatility': 0.15,
        }
        self.min_confluence_score = 0.6
        
        # Environment parameters
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        
        # Action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        
        # Initialize state
        self.terminal = False
        if len(self.df.index.unique()) > 1:
            self.unique_trade_dates = self.df.index.unique()
        else:
            self.unique_trade_dates = self.df['date'].unique()
        self.data = self.df.loc[self.day, :]
        
        # Initialize tracking variables
        self.state = self._initiate_state()
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.asset_memory = [self.initial_amount + np.sum(
            np.array(self.num_stock_shares) * np.array(self.state[1:1 + self.stock_dim])
        )]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        
        # Initialize random seed
        self._seed()

    def _sell_stock(self, index, action):
        """Execute sell action for a specific stock"""
        current_price = self.state[index + 1]
        current_holdings = self.state[index + self.stock_dim + 1]
        
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            # Sell all when turbulent
            sell_num_shares = current_holdings
        else:
            sell_num_shares = min(abs(action), current_holdings)
        
        if current_price > 0 and sell_num_shares > 0:
            sell_amount = current_price * sell_num_shares * (1 - self.sell_cost_pct[index])
            self.state[0] += sell_amount  # Add to cash
            self.state[index + self.stock_dim + 1] -= sell_num_shares
            self.cost += current_price * sell_num_shares * self.sell_cost_pct[index]
            self.trades += 1
            
        return sell_num_shares

    def _buy_stock(self, index, action):
        """Execute buy action for a specific stock"""
        current_price = self.state[index + 1]
        
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            buy_num_shares = 0
        else:
            available_amount = self.state[0] // (current_price * (1 + self.buy_cost_pct[index]))
            buy_num_shares = min(available_amount, action)
            
        if current_price > 0 and buy_num_shares > 0:
            buy_amount = current_price * buy_num_shares * (1 + self.buy_cost_pct[index])
            self.state[0] -= buy_amount  # Subtract from cash
            self.state[index + self.stock_dim + 1] += buy_num_shares
            self.cost += current_price * buy_num_shares * self.buy_cost_pct[index]
            self.trades += 1
            
        return buy_num_shares

    def step(self, actions):
        """Execute one time step within the environment"""
        self.terminal = self.day >= len(self.unique_trade_dates) - 1

        if self.terminal:
            return self._handle_terminal_step()
        else:
            return self._handle_normal_step(actions)

    def _handle_normal_step(self, actions):
        """Handle a normal (non-terminal) step in the environment"""
        actions = actions * self.hmax
        self.actions_memory.append(actions)
        
        # Calculate beginning total asset
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1:self.stock_dim + 1]) * 
            np.array(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
        )
        
        # Execute trades
        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
        
        for index in sell_index:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)
        
        for index in buy_index:
            actions[index] = self._buy_stock(index, actions[index])
        
        # Move to next state
        self.day += 1
        self.data = self.df.loc[self.day, :]
        
        # Update turbulence
        if self.turbulence_threshold is not None:
            if len(self.df.tic.unique()) == 1:
                self.turbulence = self.data[self.risk_indicator_col]
            else:
                self.turbulence = self.data[self.risk_indicator_col].values[0]
        
        # Update state
        self.state = self._update_state()
        
        # Calculate reward
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1:self.stock_dim + 1]) * 
            np.array(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
        )
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())
        
        self.reward = (end_total_asset - begin_total_asset) * self.reward_scaling
        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)
        
        return self.state, self.reward, self.terminal, False, {}

    def _handle_terminal_step(self):
        """Handle the terminal step in the environment"""
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1:self.stock_dim + 1]) * 
            np.array(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
        )
        
        if self.make_plots:
            self._make_plot()
            
        if self.episode % self.print_verbosity == 0:
            print(f"day: {self.day}, episode: {self.episode}")
            print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
            print(f"end_total_asset: {end_total_asset:0.2f}")
            print(f"total_reward: {self.rewards_memory[-1]:0.2f}")
            print(f"total_cost: {self.cost:0.2f}")
            print(f"total_trades: {self.trades}")
            print("=================================")
            
        if self.model_name != "" and self.mode != "":
            self._save_results()
            
        return self.state, self.reward, self.terminal, False, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()
        
        if self.initial:
            self.asset_memory = [self.initial_amount + np.sum(
                np.array(self.num_stock_shares) * np.array(self.state[1:1 + self.stock_dim])
            )]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1:self.stock_dim + 1]) *
                np.array(self.previous_state[self.stock_dim + 1:self.stock_dim * 2 + 1])
            )
            self.asset_memory = [previous_total_asset]
        
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        
        return self.state, {}

    def _make_plot(self):
        """Create and save performance plots"""
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Portfolio value plot
        plt.figure(figsize=(12, 4))
        plt.plot(self.date_memory, self.asset_memory, 'r')
        plt.title(f'Portfolio Value (Episode {self.episode})')
        plt.xlabel('Date')
        plt.ylabel('Asset Value')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.savefig(f'results/portfolio_value_{self.episode}.png')
        plt.close()
        
        # Returns analysis
        df_returns = pd.DataFrame(self.asset_memory, columns=['account_value'])
        df_returns['date'] = self.date_memory
        df_returns['daily_return'] = df_returns['account_value'].pct_change()
        
        plt.figure(figsize=(12, 8))
        
        # Daily returns
        plt.subplot(2, 1, 1)
        plt.plot(df_returns['date'], df_returns['daily_return'], 'b')
        plt.title('Daily Returns')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Return distribution
        plt.subplot(2, 1, 2)
        sns.histplot(df_returns['daily_return'].dropna(), bins=50, kde=True)
        plt.title('Return Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'results/returns_analysis_{self.episode}.png')
        plt.close()

    def _initiate_state(self):
        """Initialize the state space"""
        if self.initial:
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.initial_amount] +  # Cash
                    self.data.close.values.tolist() +  # Current prices
                    self.num_stock_shares +  # Stock holdings
                    sum((self.data[tech].values.tolist() 
                         for tech in self.tech_indicator_list), [])  # Technical indicators
                )
            else:
                state = (
                    [self.initial_amount] +  # Cash
                    [self.data.close] +  # Current price
                    [0] * self.stock_dim + # Stock holdings
                    sum(([self.data[tech]] for tech in self.tech_indicator_list), [])  # Technical indicators
                )
        else:
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.previous_state[0]] +  # Previous cash
                    self.data.close.values.tolist() +  # Current prices
                    self.previous_state[self.stock_dim + 1:self.stock_dim * 2 + 1] +  # Previous holdings
                    sum((self.data[tech].values.tolist() 
                         for tech in self.tech_indicator_list), [])  # Technical indicators
                )
            else:
                state = (
                    [self.previous_state[0]] +  # Previous cash
                    [self.data.close] +  # Current price
                    self.previous_state[self.stock_dim + 1:self.stock_dim * 2 + 1] +  # Previous holdings
                    sum(([self.data[tech]] for tech in self.tech_indicator_list), [])  # Technical indicators
                )
        return state

    def _update_state(self):
        """Update the state space"""
        if len(self.df.tic.unique()) > 1:
            state = (
                [self.state[0]] +  # Current cash
                self.data.close.values.tolist() +  # Current prices
                list(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1]) +  # Current holdings
                sum((self.data[tech].values.tolist() 
                     for tech in self.tech_indicator_list), [])  # Technical indicators
            )
        else:
            state = (
                [self.state[0]] +  # Current cash
                [self.data.close] +  # Current price
                list(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1]) +  # Current holdings
                sum(([self.data[tech]] for tech in self.tech_indicator_list), [])  # Technical indicators
            )
        return state

    def _get_date(self):
        """Get current date"""
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def _save_results(self):
        """Save trading results and plots"""
        # Save actions
        df_actions = pd.DataFrame(self.actions_memory)
        df_actions.columns = self.data.tic.values
        df_actions.index = self.date_memory[:-1]
        df_actions.to_csv(
            f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv"
        )
        
        # Save account value history
        df_value = pd.DataFrame({
            'date': self.date_memory,
            'account_value': self.asset_memory
        })
        df_value.to_csv(
            f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv",
            index=False
        )
        
        # Save rewards history
        df_rewards = pd.DataFrame({
            'date': self.date_memory[:-1],
            'rewards': self.rewards_memory
        })
        df_rewards.to_csv(
            f"results/rewards_{self.mode}_{self.model_name}_{self.iteration}.csv",
            index=False
        )
        
        # Calculate and save performance metrics
        df_returns = pd.DataFrame(self.asset_memory, columns=['account_value'])
        df_returns['daily_return'] = df_returns['account_value'].pct_change()
        
        stats = {
            'Initial Portfolio Value': self.asset_memory[0],
            'Final Portfolio Value': self.asset_memory[-1],
            'Total Return (%)': ((self.asset_memory[-1] - self.asset_memory[0]) / self.asset_memory[0] * 100),
            'Average Daily Return (%)': df_returns['daily_return'].mean() * 100,
            'Std Daily Return (%)': df_returns['daily_return'].std() * 100,
            'Sharpe Ratio': (252**0.5) * df_returns['daily_return'].mean() / df_returns['daily_return'].std() 
            if df_returns['daily_return'].std() != 0 else 0,
            'Total Trades': self.trades,
            'Total Trading Cost': self.cost,
            'Average Trading Cost': self.cost / self.trades if self.trades > 0 else 0
        }
        
        with open(f'results/metrics_{self.mode}_{self.model_name}_{self.iteration}.txt', 'w') as f:
            for key, value in stats.items():
                f.write(f'{key}: {value:.2f}\n')

    def render(self, mode='human'):
        """Render the environment"""
        return self.state

    def _seed(self, seed=None):
        """Set random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def save_state_memory(self):
        """Save state memory to DataFrame"""
        if len(self.df.tic.unique()) > 1:
            dates = self.date_memory[:-1]
            df_states = pd.DataFrame(
                self.state_memory,
                columns=['cash'] + 
                        [f'{tic}_price' for tic in self.df.tic.unique()] +
                        [f'{tic}_holdings' for tic in self.df.tic.unique()] +
                        self.tech_indicator_list
            )
            df_states.index = dates
        else:
            df_states = pd.DataFrame({
                'date': self.date_memory[:-1],
                'states': self.state_memory
            })
        return df_states

    def save_asset_memory(self):
        """Save asset memory to DataFrame"""
        return pd.DataFrame({
            'date': self.date_memory,
            'account_value': self.asset_memory
        })

    def get_sb_env(self):
        """Get stable-baselines env wrapper"""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
