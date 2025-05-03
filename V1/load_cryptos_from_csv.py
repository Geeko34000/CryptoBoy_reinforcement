import pandas as pd
import os
import requests

# Liste des cryptos
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # modifie selon ton besoin

# Dossier contenant les CSV
csv_folder = "data"

# Dictionnaire pour stocker les DataFrames
dataframes = {}

# Fonction pour récupérer les données depuis Binance
def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000):
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

# Chargement de chaque CSV
for symbol in symbols:
    file_path = os.path.join(csv_folder, f"{symbol}.csv")
    
    # Si le fichier n'existe pas, on le crée
    if not os.path.exists(file_path):
        print(f"[INFO] Fichier inexistant, création de {file_path}")
        open(file_path, "w").close()  # Crée un fichier vide

    # Si le fichier est vide, on le remplit avec les données Binance
    if os.path.getsize(file_path) == 0:
        print(f"[INFO] Fichier vide, chargement des données Binance pour {symbol}")
        df = fetch_binance_data(symbol)
        df.to_csv(file_path, index=False)  # Sauvegarde les nouvelles données dans le fichier
        dataframes[symbol] = df
    else:
        try:
            df = pd.read_csv(file_path)
            dataframes[symbol] = df
            print(f"[OK] Chargé : {symbol}")
        except pd.errors.EmptyDataError:
            print(f"[ERREUR] Fichier vide après création pour : {symbol}")

# Exemple d'affichage des données pour BTCUSDT
# print(dataframes["BTCUSDT"].head())
