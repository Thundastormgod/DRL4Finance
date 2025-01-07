import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import List, Dict

class StockTradingEnv(gym.Env):
    """A stock trading environment that includes VIX and turbulence for risk management"""
    
    metadata = {'render.modes': ['human']}
    
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
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        day=0,
        initial=True,
        previous_state=[],
    ):
        super().__init__()
        
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
        self.tech_indicator_list = tech_indicator_list
        
        # Risk parameters
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        
        # Calculate actual state space
        self.state_space = self._calculate_state_space()
        self.action_space_size = self.stock_dim
        
        # Risk thresholds
        self.risk_thresholds = {
            'vix_normal': 20,
            'vix_elevated': 30,
            'vix_extreme': 40,
            'turbulence_normal': 1.0,
            'turbulence_elevated': 2.0,
            'turbulence_extreme': 3.0
        }
        
        # Position sizing based on risk
        self.position_sizing = {
            'normal': 1.0,
            'elevated': 0.5,
            'extreme': 0.25,
            'crisis': 0
        }
        
        # Technical indicator weights
        self.indicator_weights = {
            'trend': {
                'sma': 0.3,
                'ema': 0.3,
                'macd': 0.4
            },
            'momentum': {
                'rsi': 0.4,
                'cci': 0.3,
                'dx': 0.3
            },
            'volatility': {
                'bbands': 1.0
            }
        }
        
        # Category weights
        self.category_weights = {
            'trend': 0.4,
            'momentum': 0.3,
            'volatility': 0.3
        }
        
        # Technical thresholds
        self.tech_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'cci_oversold': -100,
            'cci_overbought': 100,
            'bbands_squeeze': 0.1
        }
        
        # Spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_size,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        
        # Initialize state
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.state = self._initiate_state()
        self.initial = initial
        self.previous_state = previous_state
        
        # Initialize tracking
        self.asset_memory = [self.initial_amount + np.sum(
            np.array(self.num_stock_shares) * np.array(self.state[1:1 + self.stock_dim])
        )]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        
        self._initialize_trading_memory()
        self._seed()

    def _calculate_state_space(self):
        """Calculate the total dimension of the state space"""
        # Base state components: [cash] + [prices] + [shares]
        state_space = 1 + self.stock_dim + self.stock_dim
        
        # For multiple stocks
        if len(self.df.tic.unique()) > 1:
            # Add space for technical indicators (excluding vix and turbulence)
            tech_indicators = [tech for tech in self.tech_indicator_list 
                             if tech not in ['vix', 'turbulence']]
            state_space += len(tech_indicators) * self.stock_dim
        else:
            # For single stock, each technical indicator only needs one value
            tech_indicators = [tech for tech in self.tech_indicator_list 
                             if tech not in ['vix', 'turbulence']]
            state_space += len(tech_indicators)
        
        # Add space for risk indicators (these are market-wide, so only add 1 each)
        if 'vix' in self.tech_indicator_list:
            state_space += 1
        if 'turbulence' in self.tech_indicator_list:
            state_space += 1
            
        return state_space

    def _initiate_state(self):
        """Initialize the state space"""
        if len(self.df.tic.unique()) > 1:
            # Basic state components
            state = [self.initial_amount] + \
                    self.data.close.values.tolist() + \
                    self.num_stock_shares
            
            # Add technical indicators
            for tech in self.tech_indicator_list:
                if tech not in ['vix', 'turbulence']:  # Handle regular technical indicators
                    state += self.data[tech].values.tolist()
            
            # Add risk indicators at the end of state
            if 'vix' in self.tech_indicator_list:
                state.append(self.data['vix'].values[0])
            if 'turbulence' in self.tech_indicator_list:
                state.append(self.data['turbulence'].values[0])
        
        else:
            # Single stock case
            state = [self.initial_amount] + \
                    [self.data.close] + \
                    self.num_stock_shares
            
            # Add technical indicators
            for tech in self.tech_indicator_list:
                if tech not in ['vix', 'turbulence']:  # Handle regular technical indicators
                    state.append(self.data[tech])
            
            # Add risk indicators at the end of state
            if 'vix' in self.tech_indicator_list:
                state.append(self.data['vix'])
            if 'turbulence' in self.tech_indicator_list:
                state.append(self.data['turbulence'])
        
        return state
    
    def _update_state(self):
        """Update the state space"""
        if len(self.df.tic.unique()) > 1:
            # Basic state components
            state = [self.state[0]] + \
                    self.data.close.values.tolist() + \
                    list(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
            
            # Add technical indicators
            for tech in self.tech_indicator_list:
                if tech not in ['vix', 'turbulence']:  # Handle regular technical indicators
                    state += self.data[tech].values.tolist()
            
            # Add risk indicators at the end of state
            if 'vix' in self.tech_indicator_list:
                state.append(self.data['vix'].values[0])
            if 'turbulence' in self.tech_indicator_list:
                state.append(self.data['turbulence'].values[0])
        
        else:
            # Single stock case
            state = [self.state[0]] + \
                    [self.data.close] + \
                    list(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
            
            # Add technical indicators
            for tech in self.tech_indicator_list:
                if tech not in ['vix', 'turbulence']:  # Handle regular technical indicators
                    state.append(self.data[tech])
            
            # Add risk indicators at the end of state
            if 'vix' in self.tech_indicator_list:
                state.append(self.data['vix'])
            if 'turbulence' in self.tech_indicator_list:
                state.append(self.data['turbulence'])
        
        return state
    
    def get_portfolio_stats(self):
        """Calculate portfolio statistics with risk metrics"""
        df_returns = pd.DataFrame(self.asset_memory, columns=['total_assets'])
        df_returns['returns'] = df_returns['total_assets'].pct_change()
        df_returns['date'] = self.date_memory
        
        stats = {
            'Total Return': ((df_returns['total_assets'].iloc[-1] - df_returns['total_assets'].iloc[0]) / 
                           df_returns['total_assets'].iloc[0]),
            'Annual Volatility': df_returns['returns'].std() * np.sqrt(252),
            'Sharpe Ratio': (df_returns['returns'].mean() * 252) / (df_returns['returns'].std() * np.sqrt(252))
            if df_returns['returns'].std() != 0 else 0,
            'Max Drawdown': (df_returns['total_assets'] / df_returns['total_assets'].cummax() - 1).min(),
            'Total Trades': self.trades,
            'Total Cost': self.cost
        }
        
        # Add VIX and turbulence correlation if available
        if 'vix' in self.data.columns:
            vix_corr = df_returns['returns'].corr(self.df['vix'])
            stats['VIX Correlation'] = vix_corr
        
        if self.risk_indicator_col in self.data.columns:
            turb_corr = df_returns['returns'].corr(self.df[self.risk_indicator_col])
            stats['Turbulence Correlation'] = turb_corr
        
        return stats
    
    def get_trading_history(self):
        """Get detailed trading history"""
        history = pd.DataFrame({
            'date': self.date_memory[:-1],
            'actions': self.actions_memory,
            'rewards': self.rewards_memory,
            'portfolio_value': self.asset_memory[:-1],
        })
        return history
    
    def calculate_transaction_costs(self):
        """Calculate detailed transaction costs"""
        return {
            'total_cost': self.cost,
            'average_cost_per_trade': self.cost / self.trades if self.trades > 0 else 0,
            'cost_to_portfolio_value': self.cost / self.asset_memory[-1] if self.asset_memory else 0
        }
    
    def _get_date(self):
        """Get current date"""
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date
    
    def _seed(self, seed=None):
        """Set random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode != 'human':
            raise NotImplementedError(f"{mode} mode is not supported")
        
        print(f"\nDate: {self._get_date()}")
        print(f"Portfolio Value: ${self.asset_memory[-1]:,.2f}")
        print(f"Cash: ${self.state[0]:,.2f}")
        
        risk_assessment = self._assess_market_risk()
        print(f"\nRisk Assessment:")
        print(f"VIX Risk Level: {risk_assessment['vix_risk']}")
        print(f"Turbulence Risk Level: {risk_assessment['turbulence_risk']}")
        print(f"Position Size Multiplier: {risk_assessment['position_size_mult']:.2f}")
        
        print(f"\nHoldings:")
        for i in range(self.stock_dim):
            current_price = self.state[i + 1]
            holdings = self.state[i + self.stock_dim + 1]
            position_value = current_price * holdings
            print(f"Stock {i}: {holdings:.0f} shares @ ${current_price:.2f} = ${position_value:,.2f}")
        
        print(f"\nTrading Statistics:")
        print(f"Total Trades: {self.trades}")
        print(f"Total Trading Cost: ${self.cost:,.2f}")
        print("-" * 50)
        
        return self.state
    
    def get_sb_env(self):
        """Get stable-baselines environment wrapper"""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
