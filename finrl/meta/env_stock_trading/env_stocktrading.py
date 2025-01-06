from __future__ import annotations

from typing import List
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

class StockTradingEnv(gym.Env):
    """A stock trading environment that uses technical indicator confluence"""
    
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
        
        # Trading State
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.state = self._initiate_state()

        # Performance Tracking
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.asset_memory = [self.initial_amount + np.sum(
            np.array(self.num_stock_shares) * np.array(self.state[1:1+self.stock_dim])
        )]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]

        self._seed()

    def _evaluate_trend_signals(self, index, price):
        """Evaluate trend-based indicators"""
        score = 0
        total_signals = 4
        
        sma_30 = self.data['close_30_sma'].values[index] if len(self.data.shape) > 1 else self.data['close_30_sma']
        sma_60 = self.data['close_60_sma'].values[index] if len(self.data.shape) > 1 else self.data['close_60_sma']
        ema_50 = self.data['close_50_ema'].values[index] if len(self.data.shape) > 1 else self.data['close_50_ema']
        ema_200 = self.data['close_200_ema'].values[index] if len(self.data.shape) > 1 else self.data['close_200_ema']
        
        if price > sma_30: score += 1
        if price > sma_60: score += 1
        if ema_50 > ema_200: score += 1
        if sma_30 > sma_60: score += 1
        
        return score / total_signals

    def _evaluate_momentum_signals(self, index):
        """Evaluate momentum indicators"""
        score = 0
        total_signals = 3
        
        macd = self.data['macd'].values[index] if len(self.data.shape) > 1 else self.data['macd']
        dx = self.data['dx_30'].values[index] if len(self.data.shape) > 1 else self.data['dx_30']
        ema_9 = self.data['close_9_ema'].values[index] if len(self.data.shape) > 1 else self.data['close_9_ema']
        ema_12 = self.data['close_12_ema'].values[index] if len(self.data.shape) > 1 else self.data['close_12_ema']
        
        if macd > 0: score += 1
        if dx > 25: score += 1
        if ema_9 > ema_12: score += 1
        
        return score / total_signals

    def _evaluate_mean_reversion_signals(self, index, price, direction='buy'):
        """Evaluate mean reversion indicators"""
        score = 0
        total_signals = 3
        
        rsi = self.data['rsi_30'].values[index] if len(self.data.shape) > 1 else self.data['rsi_30']
        cci = self.data['cci_30'].values[index] if len(self.data.shape) > 1 else self.data['cci_30']
        sma_30 = self.data['close_30_sma'].values[index] if len(self.data.shape) > 1 else self.data['close_30_sma']
        
        if direction == 'buy':
            if rsi < self.rsi_oversold: score += 1
            if cci < -self.cci_threshold: score += 1
            if price < sma_30: score += 1
        else:
            if rsi > self.rsi_overbought: score += 1
            if cci > self.cci_threshold: score += 1
            if price > sma_30: score += 1
            
        return score / total_signals

    def _evaluate_volatility_signals(self, index, price, direction='buy'):
        """Evaluate volatility-based indicators"""
        score = 0
        total_signals = 2
        
        boll_ub = self.data['boll_ub'].values[index] if len(self.data.shape) > 1 else self.data['boll_ub']
        boll_lb = self.data['boll_lb'].values[index] if len(self.data.shape) > 1 else self.data['boll_lb']
        volatility = (boll_ub - boll_lb) / price
        
        if direction == 'buy':
            if price < boll_lb: score += 1
            if volatility < 0.02: score += 1
        else:
            if price > boll_ub: score += 1
            if volatility > 0.03: score += 1
            
        return score / total_signals

    def _get_trading_signals(self, index):
        """Calculate composite trading signals using weighted indicator scores"""
        current_price = self.state[index + 1]
        
        trend_score = self._evaluate_trend_signals(index, current_price)
        momentum_score = self._evaluate_momentum_signals(index)
        mean_rev_buy_score = self._evaluate_mean_reversion_signals(index, current_price, 'buy')
        mean_rev_sell_score = self._evaluate_mean_reversion_signals(index, current_price, 'sell')
        vol_buy_score = self._evaluate_volatility_signals(index, current_price, 'buy')
        vol_sell_score = self._evaluate_volatility_signals(index, current_price, 'sell')
        
        buy_score = (
            trend_score * self.indicator_weights['trend'] +
            momentum_score * self.indicator_weights['momentum'] +
            mean_rev_buy_score * self.indicator_weights['mean_reversion'] +
            vol_buy_score * self.indicator_weights['volatility']
        )
        
        sell_score = (
            (1 - trend_score) * self.indicator_weights['trend'] +
            (1 - momentum_score) * self.indicator_weights['momentum'] +
            mean_rev_sell_score * self.indicator_weights['mean_reversion'] +
            vol_sell_score * self.indicator_weights['volatility']
        )
        
        return {
            'buy': buy_score > self.min_confluence_score,
            'sell': sell_score > self.min_confluence_score,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'strong_buy': buy_score > 0.8,
            'strong_sell': sell_score > 0.8
        }

    def _adjust_trade_size(self, index, action, signals):
        """Adjust trade size based on signal confidence"""
        if action > 0:
            confidence_multiplier = min(1.5, max(0.5, signals['buy_score']))
            return action * confidence_multiplier
        elif action < 0:
            confidence_multiplier = min(1.5, max(0.5, signals['sell_score']))
            return action * confidence_multiplier
        return action

    def _buy_stock(self, index, action):
        """Execute buy orders with technical analysis confirmation"""
        signals = self._get_trading_signals(index)
        adjusted_action = self._adjust_trade_size(index, action, signals)
        
        def _do_buy():
            if self.state[index + 2 * self.stock_dim + 1] != True:
                available_amount = self.state[0] // (self.state[index + 1] * (1 + self.buy_cost_pct[index]))
                
                if signals['buy'] or signals['strong_buy'] or adjusted_action > self.hmax * 0.8:
                    buy_num_shares = min(available_amount, adjusted_action)
                    buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index])
                    self.state[0] -= buy_amount
                    self.state[index + self.stock_dim + 1] += buy_num_shares
                    self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                    self.trades += 1
                    return buy_num_shares
            return 0

        if self.turbulence_threshold is None:
            return _do_buy()
        elif self.turbulence < self.turbulence_threshold:
            return _do_buy()
        return 0

    def _sell_stock(self, index, action):
        """Execute sell orders with technical analysis confirmation"""
        signals = self._get_trading_signals(index)
        adjusted_action = self._adjust_trade_size(index, action, signals)
        
        def _do_sell_normal():
            if self.state[index + 2 * self.stock_dim + 1] != True:
                if self.state[index + self.stock_dim + 1] > 0:
                    if signals['sell'] or signals['strong_sell'] or abs(adjusted_action) > self.hmax * 0.8:
                        sell_num_shares = min(abs(adjusted_action), self.state[index + self.stock_dim + 1])
                        sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] -= sell_num_shares
                        self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                        self.trades += 1
                        return sell_num_shares
            return 0

        if self.turbulence_threshold is None:
            return _do_sell_normal()
        elif self.turbulence >= self.turbulence_threshold:
            if self.state[index + self.stock_dim + 1] > 0:
                sell_num_shares = self.state[index + self.stock_dim + 1]
                sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                self.state[0] += sell_amount
                self.state[index + self.stock_dim + 1] = 0
                self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                self.trades += 1
                return sell_num_shares
            return 0
        else:
            return _do_sell_normal()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            if self.make_plots:
                self._make_plot()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)]) *
                np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            
            if df_total_value["daily_return"].std() != 0:
                sharpe = (252**0.5) * df_total_value["daily_return"].mean() / df_total_value["daily_return"].std()
            
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {end_total_asset - self.asset_memory[0]:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                print(f"Sharpe: {sharpe:0.2f}")
                print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * self.hmax  # scale up actions
            self.actions_memory.append(actions)
            
            # Calculate beginning total asset
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)]) *
                np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )

            # Update price and turbulence
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.turbulence = self.data[self.risk_indicator_col] if self.risk_indicator_col in self.data.index else 0
            self.state = self._update_state()

            # Execute trades
            for index, action in enumerate(actions):
                if action > 0:  # buy
                    bought_shares = self._buy_stock(index, action)
                else:  # sell
                    sold_shares = self._sell_stock(index, action)

            # Calculate end total asset
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)]) *
                np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )

            # Calculate reward
            self.reward = end_total_asset - begin_total_asset
            self.reward = self.reward * self.reward_scaling
            self.rewards_memory.append(self.reward)
            
            # Update memory
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.state_memory.append(self.state)

            return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()
        
        # Reset performance tracking
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.asset_memory = [self.initial_amount + np.sum(
            np.array(self.num_stock_shares) * np.array(self.state[1:1+self.stock_dim])
        )]
        self.episode += 1

        return self.state

    def render(self, mode='human'):
        return self.state

    def _initiate_state(self):
        if len(self.df.shape) > 1:
            # Get prices and technical indicators
            prices = self.df.loc[self.day, [f"close{i+1}" for i in range(self.stock_dim)]].values
            tech_indicators = self.df.loc[self.day, self.tech_indicator_list].values
            
            # Initialize holdings if first episode
            if self.initial:
                holdings = np.array(self.num_stock_shares)
            else:
                holdings = self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]

        else:
            prices = self.df[[f"close{i+1}" for i in range(self.stock_dim)]].values
            tech_indicators = self.df[self.tech_indicator_list].values
            if self.initial:
                holdings = np.array(self.num_stock_shares)
            else:
                holdings = self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]

        state = np.hstack([
            [self.state[0]] if self.initial else [self.previous_state[0]],  # cash
            prices,  # prices
            holdings,  # holdings
            [False] * self.stock_dim  # flags for sell cooldown
        ])
        state = np.concatenate((state, tech_indicators))
        return state

    def _update_state(self):
        if len(self.df.shape) > 1:
            prices = self.df.loc[self.day, [f"close{i+1}" for i in range(self.stock_dim)]].values
            tech_indicators = self.df.loc[self.day, self.tech_indicator_list].values
        else:
            prices = self.df[[f"close{i+1}" for i in range(self.stock_dim)]].values
            tech_indicators = self.df[self.tech_indicator_list].values

        state = np.hstack([
            [self.state[0]],  # cash
            prices,  # prices
            self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)],  # holdings
            self.state[(self.stock_dim * 2 + 1):(self.stock_dim * 3 + 1)]  # flags
        ])
        state = np.concatenate((state, tech_indicators))
        return state

    def _get_date(self):
        if len(self.df.shape) > 1:
            return self.df.index[self.day]
        else:
            return self.df.index

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _make_plot(self):
        plt.plot(self.asset_memory, 'r')
        plt.savefig(f'results/account_value_trade_{self.episode}.png')
        plt.close()
