import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FinTechTradingEnv(gym.Env):
    """A sophisticated stock trading environment for reinforcement learning"""
    
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
        """Initialize the trading environment with given parameters"""
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
        
        # Trading parameters
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
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
        
        # Technical Analysis Parameters
        self.indicator_weights = {
            'trend': 0.35,
            'momentum': 0.25,
            'mean_reversion': 0.25,
            'volatility': 0.15,
        }
        self.min_confluence_score = 0.6
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.cci_threshold = 100
        
        # Initialize trading state
        self.unique_trade_dates = self.df['date'].unique()
        self.data = self.df.loc[self.df['date'] == self.unique_trade_dates[self.day]]
        self.terminal = False
        
        # Initialize tracking variables
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.trades = 0
        self.episode = 0
        self.reward = 0
        self.cost = 0
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        
        self.state = self._initiate_state()

    def _get_stock_price(self, tic):
        """Get the current stock price for a given ticker"""
        return self.data[self.data['tic'] == tic]['close'].values[0]
    
    def _get_stock_indicators(self, tic):
        """Get the current technical indicators for a given ticker"""
        stock_data = self.data[self.data['tic'] == tic]
        return [stock_data[indicator].values[0] for indicator in self.tech_indicator_list]

    def _initiate_state(self):
        """Initialize the state space"""
        unique_tickers = self.df['tic'].unique()
        
        if self.initial:
            # Start with initial portfolio
            state = [self.initial_amount]  # Cash
            
            # Add current prices and holdings for each stock
            for i, tic in enumerate(unique_tickers):
                state.append(self._get_stock_price(tic))  # Current price
                state.append(self.num_stock_shares[i])    # Number of shares
            
            # Add technical indicators for each stock
            for tic in unique_tickers:
                state.extend(self._get_stock_indicators(tic))
                
        else:
            # Use previous state
            state = self.previous_state
            
        return np.array(state)

    def _update_state(self):
        """Update the state space on each new day"""
        unique_tickers = self.df['tic'].unique()
        
        state = [self.state[0]]  # Keep current cash
        
        # Update prices and holdings for each stock
        n_stocks = len(unique_tickers)
        holdings = self.state[n_stocks + 1:2*n_stocks + 1]  # Extract current holdings
        
        for i, tic in enumerate(unique_tickers):
            state.append(self._get_stock_price(tic))  # Current price
            state.append(holdings[i])                 # Keep current holdings
            
        # Update technical indicators for each stock
        for tic in unique_tickers:
            state.extend(self._get_stock_indicators(tic))
            
        return np.array(state)

    def _get_date(self):
        """Get the current date"""
        return self.data['date'].iloc[0]

    def _calculate_reward(self, begin_total_asset, end_total_asset):
        """Calculate step reward"""
        reward = end_total_asset - begin_total_asset
        reward = reward * self.reward_scaling
        return reward

    def _make_plot(self):
        """Create and save various trading performance plots"""
        # Create a directory for results if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Account value over time
        plt.figure(figsize=(12, 4))
        plt.plot(self.date_memory, self.asset_memory, 'r', label='Portfolio Value')
        plt.title(f'Portfolio Value Over Time (Episode {self.episode})')
        plt.xlabel('Date')
        plt.ylabel('Account Value ($)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/account_value_trade_{self.episode}.png')
        plt.close()
        
        # Daily returns analysis
        df_returns = pd.DataFrame(self.asset_memory, columns=['account_value'])
        df_returns['date'] = self.date_memory
        df_returns['daily_return'] = df_returns['account_value'].pct_change()
        
        plt.figure(figsize=(12, 8))
        
        # Daily returns subplot
        plt.subplot(2, 1, 1)
        plt.plot(df_returns['date'], df_returns['daily_return'], 'b', label='Daily Returns')
        plt.title(f'Daily Returns (Episode {self.episode})')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        
        # Return distribution subplot
        plt.subplot(2, 1, 2)
        sns.histplot(df_returns['daily_return'].dropna(), bins=50, kde=True)
        plt.title('Return Distribution')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/returns_analysis_{self.episode}.png')
        plt.close()
        
        # Trade analysis
        if len(self.actions_memory) > 0:
            df_actions = pd.DataFrame(self.actions_memory)
            df_actions.columns = [f'Stock_{i+1}' for i in range(self.stock_dim)]
            
            plt.figure(figsize=(12, 6))
            
            # Plot trading actions heatmap
            sns.heatmap(df_actions.T, cmap='RdYlGn', center=0, 
                       xticklabels=False, yticklabels=True)
            plt.title(f'Trading Actions Heatmap (Episode {self.episode})')
            plt.xlabel('Time Step')
            plt.ylabel('Stock')
            
            plt.tight_layout()
            plt.savefig(f'results/trading_actions_{self.episode}.png')
            plt.close()
            
            # Save trading statistics
            stats = {
                'Initial Portfolio Value': self.asset_memory[0],
                'Final Portfolio Value': self.asset_memory[-1],
                'Total Return': (self.asset_memory[-1] - self.asset_memory[0]) / self.asset_memory[0] * 100,
                'Total Trades': self.trades,
                'Total Trading Cost': self.cost,
                'Sharpe Ratio': (252**0.5) * df_returns['daily_return'].mean() / df_returns['daily_return'].std() 
                if df_returns['daily_return'].std() != 0 else 0,
                'Max Drawdown': (df_returns['account_value'].max() - df_returns['account_value'].min()) / df_returns['account_value'].max() * 100
            }
            
            with open(f'results/trading_stats_{self.episode}.txt', 'w') as f:
                for key, value in stats.items():
                    f.write(f'{key}: {value:.2f}\n')

    def step(self, actions):
        """Execute one time step within the environment"""
        self.terminal = self.day >= len(self.unique_trade_dates) - 1

        if self.terminal:
            # Calculate final portfolio value
            end_total_asset = self.state[0] + sum(
                [self.state[i + 1] * self.state[i + self.stock_dim + 1] 
                 for i in range(self.stock_dim)]
            )
            
            # Create plots if enabled
            if self.make_plots:
                self._make_plot()
            
            # Print episode summary
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {self.rewards_memory[-1]:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            # Scale actions
            actions = actions * self.hmax
            self.actions_memory.append(actions)
            
            # Calculate beginning total asset
            begin_total_asset = self.state[0] + sum(
                [self.state[i + 1] * self.state[i + self.stock_dim + 1] 
                 for i in range(self.stock_dim)]
            )
            
            # Execute trades
            unique_tickers = self.df['tic'].unique()
            for i, action in enumerate(actions):
                current_price = self._get_stock_price(unique_tickers[i])
                current_holdings = self.state[i + self.stock_dim + 1]
                
                if action > 0:  # buy
                    # Calculate maximum shares possible to buy
                    max_possible = self.state[0] // (current_price * (1 + self.buy_cost_pct[i]))
                    buy_num_shares = min(max_possible, action)
                    
                    if buy_num_shares > 0:
                        buy_amount = current_price * buy_num_shares * (1 + self.buy_cost_pct[i])
                        self.state[0] -= buy_amount  # Reduce cash
                        self.state[i + self.stock_dim + 1] += buy_num_shares  # Increase holdings
                        self.cost += current_price * buy_num_shares * self.buy_cost_pct[i]
                        self.trades += 1
                        
                elif action < 0:  # sell
                    sell_num_shares = min(abs(action), current_holdings)
                    
                    if sell_num_shares > 0:
                        sell_amount = current_price * sell_num_shares * (1 - self.sell_cost_pct[i])
                        self.state[0] += sell_amount  # Increase cash
                        self.state[i + self.stock_dim + 1] -= sell_num_shares  # Reduce holdings
                        self.cost += current_price * sell_num_shares * self.sell_cost_pct[i]
                        self.trades += 1
            
            # Move to next day
            self.day += 1
            self.data = self.df.loc[self.df['date'] == self.unique_trade_dates[self.day]]
            
            # Update state
            self.state = self._update_state()
            
            # Calculate reward
            end_total_asset = self.state[0] + sum(
                [self.state[i + 1] * self.state[i + self.stock_dim + 1] 
                 for i in range(self.stock_dim)]
            )
            self.reward = self._calculate_reward(begin_total_asset, end_total_asset)
            self.rewards_memory.append(self.reward)
            
            # Update memory
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.state_memory.append(self.state)

            return self.state, self.reward, self.terminal, {}

    def reset(self):
        """Reset the environment to initial state"""
        self.day = 0
        self.data = self.df.loc[self.df['date'] == self.unique_trade_dates[self.day]]
        self.state = self._initiate_state()
        
        # Reset performance tracking
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.trades = 0
        self.cost = 0
        self.terminal = False
        self.episode += 1

        return self.state

    def render(self, mode='human'):
        """Render the environment"""
        return self.state

    def _seed(self, seed=None):
        """Set random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
