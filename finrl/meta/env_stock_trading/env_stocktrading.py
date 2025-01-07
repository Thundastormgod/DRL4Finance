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
        state_space: int,
        action_space: int,
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
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        
        # Risk parameters
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        
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
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
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
    
    def _assess_market_risk(self) -> dict:
        """Evaluate market risk levels using VIX and turbulence"""
        current_vix = self.data['vix'].values[0] if 'vix' in self.data.columns else None
        current_turbulence = self.data[self.risk_indicator_col].values[0] if self.risk_indicator_col in self.data.columns else None
        
        risk_assessment = {
            'vix_risk': 'normal',
            'turbulence_risk': 'normal',
            'position_size_mult': 1.0,
            'should_exit': False
        }
        
        # Assess VIX risk level
        if current_vix is not None:
            if current_vix >= self.risk_thresholds['vix_extreme']:
                risk_assessment['vix_risk'] = 'extreme'
            elif current_vix >= self.risk_thresholds['vix_elevated']:
                risk_assessment['vix_risk'] = 'elevated'
        
        # Assess turbulence risk level
        if current_turbulence is not None:
            if current_turbulence >= self.risk_thresholds['turbulence_extreme']:
                risk_assessment['turbulence_risk'] = 'extreme'
            elif current_turbulence >= self.risk_thresholds['turbulence_elevated']:
                risk_assessment['turbulence_risk'] = 'elevated'
        
        # Determine overall risk level and position sizing
        if risk_assessment['vix_risk'] == 'extreme' or risk_assessment['turbulence_risk'] == 'extreme':
            risk_assessment['position_size_mult'] = self.position_sizing['extreme']
            risk_assessment['should_exit'] = True
        elif risk_assessment['vix_risk'] == 'elevated' or risk_assessment['turbulence_risk'] == 'elevated':
            risk_assessment['position_size_mult'] = self.position_sizing['elevated']
        
        return risk_assessment
    
    def _calculate_technical_signals(self, index: int) -> Dict[str, float]:
        """Calculate technical signals including risk indicators"""
        signals = {
            'trend': 0,
            'momentum': 0,
            'volatility': 0
        }
        
        # Trend signals
        if 'close_30_sma' in self.tech_indicator_list and 'close_60_sma' in self.tech_indicator_list:
            sma_signal = 1 if self.data['close_30_sma'].values[index] > self.data['close_60_sma'].values[index] else -1
            signals['trend'] += sma_signal * self.indicator_weights['trend']['sma']
        
        if all(x in self.tech_indicator_list for x in ['close_9_ema', 'close_50_ema']):
            ema_signal = 1 if self.data['close_9_ema'].values[index] > self.data['close_50_ema'].values[index] else -1
            signals['trend'] += ema_signal * self.indicator_weights['trend']['ema']
        
        if 'macd' in self.tech_indicator_list:
            macd_signal = 1 if self.data['macd'].values[index] > 0 else -1
            signals['trend'] += macd_signal * self.indicator_weights['trend']['macd']
        
        # Momentum signals
        if 'rsi_30' in self.tech_indicator_list:
            rsi = self.data['rsi_30'].values[index]
            if rsi < self.tech_thresholds['rsi_oversold']:
                rsi_signal = 1
            elif rsi > self.tech_thresholds['rsi_overbought']:
                rsi_signal = -1
            else:
                rsi_signal = 0
            signals['momentum'] += rsi_signal * self.indicator_weights['momentum']['rsi']
        
        if 'cci_30' in self.tech_indicator_list:
            cci = self.data['cci_30'].values[index]
            if cci < self.tech_thresholds['cci_oversold']:
                cci_signal = 1
            elif cci > self.tech_thresholds['cci_overbought']:
                cci_signal = -1
            else:
                cci_signal = 0
            signals['momentum'] += cci_signal * self.indicator_weights['momentum']['cci']
        
        if 'dx_30' in self.tech_indicator_list:
            dx = self.data['dx_30'].values[index]
            dx_signal = 1 if dx > 25 else (-1 if dx < -25 else 0)
            signals['momentum'] += dx_signal * self.indicator_weights['momentum']['dx']
        
        # Volatility signals
        if all(x in self.tech_indicator_list for x in ['boll_ub', 'boll_lb']):
            current_price = self.data['close'].values[index]
            upper_band = self.data['boll_ub'].values[index]
            lower_band = self.data['boll_lb'].values[index]
            
            if current_price < lower_band:
                bb_signal = 1
            elif current_price > upper_band:
                bb_signal = -1
            else:
                bb_signal = 0
            signals['volatility'] += bb_signal * self.indicator_weights['volatility']['bbands']
        
        return signals
    
    def _calculate_composite_signal(self, signals: Dict[str, float]) -> float:
        """Calculate overall trading signal"""
        composite_signal = 0
        for category, weight in self.category_weights.items():
            composite_signal += signals[category] * weight
        return np.clip(composite_signal, -1, 1)
    
    def _modify_action(self, action: float, index: int) -> float:
        """Modify action based on technical and risk signals"""
        signals = self._calculate_technical_signals(index)
        technical_signal = self._calculate_composite_signal(signals)
        risk_assessment = self._assess_market_risk()
        
        if risk_assessment['should_exit']:
            return -1.0
        
        modified_action = 0.7 * action + 0.3 * technical_signal
        modified_action *= risk_assessment['position_size_mult']
        
        return np.clip(modified_action, -1, 1)
    
    def _initialize_trading_memory(self):
        """Initialize trading tracking variables"""
        self.cost = 0
        self.trades = 0
        self.reward = 0
        self.turbulence = 0
        self.current_position_duration = np.zeros(self.stock_dim)
        self.position_history = []
    
    def _buy_stock(self, index: int, action: float) -> float:
        """Execute buy order with risk management"""
        modified_action = self._modify_action(action, index)
        risk_assessment = self._assess_market_risk()
        
        if risk_assessment['should_exit']:
            return 0
        
        if modified_action > 0:
            current_price = self.state[index + 1]
            available_amount = self.state[0] // (current_price * (1 + self.buy_cost_pct[index]))
            buy_num_shares = min(
                available_amount,
                modified_action * self.hmax * risk_assessment['position_size_mult']
            )
            
            if buy_num_shares > 0:
                buy_amount = current_price * buy_num_shares * (1 + self.buy_cost_pct[index])
                self.state[0] -= buy_amount
                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.cost += current_price * buy_num_shares * self.buy_cost_pct[index]
                self.trades += 1
                return buy_num_shares
        
        return 0
    
    def _sell_stock(self, index: int, action: float) -> float:
        """Execute sell order with risk management"""
        modified_action = self._modify_action(action, index)
        risk_assessment = self._assess_market_risk()
        
        if risk_assessment['should_exit']:
            modified_action = -1.0
        
        if modified_action < 0:
            current_price = self.state[index + 1]
            current_holdings = self.state[index + self.stock_dim + 1]
            
            sell_num_shares = current_holdings if risk_assessment['should_exit'] else min(
                abs(modified_action) * self.hmax,
                current_holdings
            )
            
            if sell_num_shares > 0:
                sell_amount = current_price * sell_num_shares * (1 - self.sell_cost_pct[index])
                self.state[0] += sell_amount
                self.state[index + self.stock_dim + 1] -= sell_num_shares
                self.cost += current_price * sell_num_shares * self.sell_cost_pct[index]
                self.trades += 1
                return sell_num_shares
        
        return 0
    
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        
        if self.terminal:
            return self.state, self.reward, self.terminal, False, {}
        
        else:
            actions = actions * self.hmax
            self.actions_memory.append(actions)
            
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1:self.stock_dim + 1]) *
                np.array(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
            )
            
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
            
            for index in sell_index:
                self._sell_stock(index, actions[index])
            
            for index in buy_index:
                self._buy_stock(index, actions[index])
            
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.state = self._update_state()
            
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:self.stock_dim + 1]) *
                np.array(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
            )
            
            self.reward = self.reward_scaling * (end_total_asset - begin_total_asset)
            self.reward *= max(0, 1 - self.cost / end_total_asset)
            
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            
            return self.state, self.reward, self.terminal, False, {}
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
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
        
        self._initialize_trading_memory()
        
        return self.state, {}
    
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
