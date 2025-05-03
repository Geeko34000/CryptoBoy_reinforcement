# -*- coding: utf-8 -*-
"""
Created on Sat May  3 14:03:02 2025

@author: aknin
"""

import gym
import ta
import numpy as np
import pandas as pd

class CryptoTradingEnv(gym.Env):
    def __init__(self, df, fee=0.001):
        super().__init__()
        self.df = df
        self.df['rsi'] = ta.momentum.RSIIndicator(close=self.df['close']).rsi()
        self.df['macd'] = ta.trend.MACD(close=self.df['close']).macd()
        self.df['sma'] = ta.trend.SMAIndicator(close=self.df['close'], window=14).sma_indicator()
        self.df = self.df.dropna().reset_index(drop=True)
        self.fee = fee
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
    
    def reset(self):
        self.balance = 1000
        self.crypto = 0
        self.step_idx = 0
        return self._get_obs()
    
    def step(self, action):
        price = self.df.iloc[self.step_idx]['close']
        action = float(action[0])

        # Trade
        trade_amount = action * self.balance
        if action > 0:
            bought = (trade_amount * (1 - self.fee)) / price
            self.crypto += bought
            self.balance -= trade_amount
        elif action < 0:
            sold = min(-action * self.crypto, self.crypto)
            self.crypto -= sold
            self.balance += sold * price * (1 - self.fee)
        
        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 1
        new_price = self.df.iloc[self.step_idx]['close']
        value = self.balance + self.crypto * new_price
        reward = value - (self.balance + self.crypto * price)

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        row = self.df.iloc[self.step_idx]
        return np.array([
            row['close'],
            row['rsi'],
            row['macd'],
            row['sma'],
            self.balance,
            self.crypto
        ])

