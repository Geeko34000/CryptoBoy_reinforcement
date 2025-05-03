import gymnasium as gym
import ta
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class CryptoTradingEnv(gym.Env):
    """
    Environnement de trading crypto amélioré avec plus d'indicateurs techniques
    et un système de récompense plus sophistiqué.
    """
    
    def __init__(self, df: pd.DataFrame, window_size: int = 20, fee: float = 0.001, 
                 initial_balance: float = 1000.0, reward_scaling: float = 0.01):
        super().__init__()
        
        # Prétraitement et calcul des indicateurs
        self.raw_df = df.copy()
        self.df = self._preprocess_data(df)
        self.window_size = window_size
        self.fee = fee
        self.initial_balance = initial_balance
        self.reward_scaling = reward_scaling
        
        # Définition des espaces d'action et d'observation
        # Action: [-1, 1] où -1 = vendre tout, 0 = hold, 1 = acheter tout
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation: prix + indicateurs + état du portefeuille + historique des prix
        n_features = self.df.shape[1] - 1  # -1 pour exclure la colonne timestamp
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features + 2 + window_size,), dtype=np.float32
        )
        
        # Variables d'état
        self.step_idx = 0
        self.balance = initial_balance
        self.crypto = 0.0
        self.previous_net_worth = initial_balance
        self.net_worth_history = []
        self.trades = []
        
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs techniques nécessaires."""
        processed = df.copy()
        
        # Normalisation des prix
        for col in ['open', 'high', 'low', 'close']:
            if col in processed.columns:
                processed[col] = processed[col].astype(float)
        
        # Conversion des timestamps si nécessaire
        if 'timestamp' in processed.columns and not pd.api.types.is_datetime64_any_dtype(processed['timestamp']):
            processed['timestamp'] = pd.to_datetime(processed['timestamp'])
        
        # Indicateurs de tendance
        processed['sma_20'] = ta.trend.SMAIndicator(close=processed['close'], window=20).sma_indicator()
        processed['sma_50'] = ta.trend.SMAIndicator(close=processed['close'], window=50).sma_indicator()
        processed['sma_ratio'] = processed['sma_20'] / processed['sma_50']
        processed['ema_12'] = ta.trend.EMAIndicator(close=processed['close'], window=12).ema_indicator()
        processed['ema_26'] = ta.trend.EMAIndicator(close=processed['close'], window=26).ema_indicator()
        
        # MACD
        macd = ta.trend.MACD(close=processed['close'])
        processed['macd'] = macd.macd()
        processed['macd_signal'] = macd.macd_signal()
        processed['macd_diff'] = macd.macd_diff()
        
        # Momentum
        processed['rsi'] = ta.momentum.RSIIndicator(close=processed['close']).rsi()
        processed['stoch'] = ta.momentum.StochasticOscillator(
            high=processed['high'], low=processed['low'], close=processed['close']
        ).stoch()
        
        # Volatilité
        processed['bbands_high'] = ta.volatility.BollingerBands(
            close=processed['close'], window=20
        ).bollinger_hband()
        processed['bbands_low'] = ta.volatility.BollingerBands(
            close=processed['close'], window=20
        ).bollinger_lband()
        processed['bbands_width'] = (processed['bbands_high'] - processed['bbands_low']) / processed['close']
        
        # Rendements
        processed['returns'] = processed['close'].pct_change()
        processed['log_returns'] = np.log(processed['close'] / processed['close'].shift(1))
        
        # Supprimer les lignes avec des NaN et réinitialiser l'index
        return processed.dropna().reset_index(drop=True)
    
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Réinitialise l'environnement et retourne l'observation initiale."""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.crypto = 0.0
        self.step_idx = self.window_size
        self.previous_net_worth = self.initial_balance
        self.net_worth_history = [self.initial_balance]
        self.trades = []
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: Valeur entre -1 et 1 indiquant la proportion de capital à allouer
                  -1 = vendre tout, 0 = hold, 1 = acheter tout
        """
        # Contraindre l'action à l'intervalle valide
        action = np.clip(action, -1, 1)[0]
        
        # Prix actuel
        current_price = self.df.iloc[self.step_idx]['close']
        
        # Calculer la valeur nette avant l'action
        previous_net_worth = self.balance + self.crypto * current_price
        
        # Exécuter l'action de trading
        if action > 0:  # Acheter
            buy_amount = self.balance * action
            buy_crypto = (buy_amount * (1 - self.fee)) / current_price
            
            if buy_amount > 0:
                self.trades.append({
                    'step': self.step_idx,
                    'price': current_price,
                    'type': 'buy',
                    'amount': buy_crypto,
                    'cost': buy_amount
                })
                
                self.crypto += buy_crypto
                self.balance -= buy_amount
                
        elif action < 0:  # Vendre
            sell_crypto = self.crypto * abs(action)
            sell_amount = sell_crypto * current_price * (1 - self.fee)
            
            if sell_crypto > 0:
                self.trades.append({
                    'step': self.step_idx,
                    'price': current_price,
                    'type': 'sell',
                    'amount': sell_crypto,
                    'revenue': sell_amount
                })
                
                self.crypto -= sell_crypto
                self.balance += sell_amount
        
        # Avancer d'un pas de temps
        self.step_idx += 1
        
        # Vérifier si la simulation est terminée
        done = self.step_idx >= len(self.df) - 1
        truncated = False
        
        # Calculer la nouvelle valeur nette
        new_price = self.df.iloc[self.step_idx]['close']
        current_net_worth = self.balance + self.crypto * new_price
        self.net_worth_history.append(current_net_worth)
        
        # Calculer la récompense
        reward = self._calculate_reward(previous_net_worth, current_net_worth)
        
        # Mettre à jour la valeur nette précédente
        self.previous_net_worth = current_net_worth
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _calculate_reward(self, previous_net_worth: float, current_net_worth: float) -> float:
        """Calcule la récompense basée sur le changement de valeur nette et d'autres facteurs."""
        # Récompense principale: changement relatif de la valeur nette
        pct_change = (current_net_worth - previous_net_worth) / previous_net_worth
        reward = pct_change * self.reward_scaling
        
        # Ajouter des pénalités pour des actions qui peuvent être nocives
        current_price = self.df.iloc[self.step_idx]['close']
        
        # Pénalité pour être trop exposé au marché en période de forte volatilité
        volatility = self.df.iloc[self.step_idx]['bbands_width']
        if volatility > 0.05 and self.crypto * current_price > 0.8 * current_net_worth:
            reward -= 0.001
        
        # Pénalité pour inactivité prolongée (pas de trades)
        if len(self.trades) == 0 and self.step_idx > self.window_size + 20:
            reward -= 0.0005
            
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Construit et retourne l'observation actuelle."""
        # État actuel des indicateurs
        row = self.df.iloc[self.step_idx]
        
        # Extraire toutes les colonnes numériques sauf 'timestamp'
        features = row.drop('timestamp' if 'timestamp' in row.index else []).values
        
        # Ajouter état du portefeuille
        current_price = row['close']
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Balance normalisée
            self.crypto * current_price / self.initial_balance  # Valeur crypto normalisée
        ])
        
        # Historique des prix
        price_history = np.array([
            self.df.iloc[self.step_idx - self.window_size + i]['close'] / current_price - 1
            for i in range(self.window_size)
        ])
        
        # Combiner tout
        return np.concatenate([features, portfolio_state, price_history]).astype(np.float32)
    
    def render(self):
        """Pas d'affichage en temps réel nécessaire pour le moment."""
        pass