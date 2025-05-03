import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import CryptoTradingEnv

# Charger les données historiques (les mêmes que pour l'entraînement, ou nouvelles)
df = pd.read_csv("data/ETHUSDT.csv")

# Charger le modèle entraîné
model = PPO.load("ppo_trader")

# Recréer l'environnement
env = CryptoTradingEnv(df)

obs = env.reset()
done = False

portfolio_values = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    current_price = df.iloc[env.step_idx]["close"]
    total_value = env.balance + env.crypto * current_price
    portfolio_values.append(total_value)

# Tracer l’évolution du capital
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label="Capital total ($)")
plt.title("Évolution du portefeuille pendant le backtest")
plt.xlabel("Temps (pas)")
plt.ylabel("Valeur totale ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
