from stable_baselines3 import PPO
from env import CryptoTradingEnv
from IPython.display import display
import ipywidgets as widgets
import requests
import pandas as pd
import threading

# === 1. Charger les données ===
def load_data(symbol="BTCUSDT", interval="1h", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["symbol"] = symbol
    return df

# === 2. Setup bouton d'arrêt ===
stop_training = threading.Event()

button = widgets.Button(description="Arrêter et enregistrer")
def on_click(b):
    stop_training.set()
button.on_click(on_click)
display(button)

# === 3. Callback d'enregistrement ===
class SaveOnClickCallback:
    def __init__(self, model):
        self.model = model

    def __call__(self, _locals, _globals):
        if stop_training.is_set():
            print(">> Arrêt demandé. Sauvegarde du modèle...")
            self.model.save("ppo_trader_manual")
            return False
        return True

# === 4. Initialisation et entraînement ===
df = load_data()
env = CryptoTradingEnv(df)
model = PPO("MlpPolicy", env, verbose=1)

callback = SaveOnClickCallback(model)
model.learn(total_timesteps=500_000, callback=callback)

model.save("ppo_trader")
