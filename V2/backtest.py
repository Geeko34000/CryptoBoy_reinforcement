import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import os
from datetime import datetime

from agents import create_agent, TradingAgent
from env import CryptoTradingEnv
from metrics import TradingMetrics

class Backtester:
    """
    Classe pour backtester des stratégies de trading.
    """
    
    def __init__(self, df: pd.DataFrame, window_size: int = 20, initial_balance: float = 1000.0,
                fee: float = 0.001, reward_scaling: float = 0.01):
        """
        Initialise le backtester.
        
        Args:
            df: DataFrame contenant les données historiques
            window_size: Taille de la fenêtre d'observation
            initial_balance: Capital initial pour le backtest
            fee: Frais de transaction
            reward_scaling: Facteur d'échelle pour les récompenses
        """
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee = fee
        self.reward_scaling = reward_scaling
        
        # Créer l'environnement
        self.env = CryptoTradingEnv(
            df=df,
            window_size=window_size,
            fee=fee,
            initial_balance=initial_balance,
            reward_scaling=reward_scaling
        )
        
        # Métriques pour l'évaluation
        self.metrics = TradingMetrics()
        
        # Stockage des résultats
        self.results = None
    
    def run_backtest(self, agent: TradingAgent, render: bool = False) -> Dict:
        """
        Exécute un backtest avec l'agent spécifié.
        
        Args:
            agent: Agent de trading à tester
            render: Afficher les graphiques pendant le backtest
        
        Returns:
            Dictionnaire contenant les résultats du backtest
        """
        # Réinitialiser l'environnement
        obs, _ = self.env.reset()
        done = False
        truncated = False
        
        # Variables de suivi
        portfolio_values = []
        balances = []
        crypto_holdings = []
        actions_taken = []
        rewards = []
        timestamps = []
        
        print("Démarrage du backtest...")
        
        # Boucle principale de backtest
        while not done and not truncated:
            # Prédire l'action
            action = agent.predict(obs)
            
            # Exécuter l'action
            new_obs, reward, done, truncated, info = self.env.step(action)
            
            # Mettre à jour l'observation
            obs = new_obs
            
            # Récupérer les informations de l'épisode
            current_step = self.env.step_idx
            current_price = self.df.iloc[current_step]['close']
            current_timestamp = self.df.iloc[current_step]['timestamp']
            total_value = self.env.balance + self.env.crypto * current_price
            
            # Stocker les données
            portfolio_values.append(total_value)
            balances.append(self.env.balance)
            crypto_holdings.append(self.env.crypto)
            actions_taken.append(float(action[0]))
            rewards.append(reward)
            timestamps.append(current_timestamp)
        
        print("Backtest terminé.")
        
        # Compiler les résultats
        self.results = {
            'portfolio_values': portfolio_values,
            'balances': balances,
            'crypto_holdings': crypto_holdings,
            'actions': actions_taken,
            'rewards': rewards,
            'timestamps': timestamps,
            'prices': self.df.iloc[self.window_size:self.window_size + len(portfolio_values)]['close'].values,
            'trades': self.env.trades
        }
        
        # Calculer les métriques
        metrics = self.metrics.calculate_metrics(self.results, self.initial_balance)
        self.results.update(metrics)
        
        return self.results
    
    def plot_results(self, save_path: Optional[str] = None, 
                    show_plots: bool = True) -> None:
        """
        Génère des graphiques d'analyse du backtest.
        
        Args:
            save_path: Chemin où sauvegarder les graphiques (None pour ne pas sauvegarder)
            show_plots: Afficher les graphiques
        """
        if self.results is None:
            print("Aucun résultat de backtest à afficher.")
            return
        
        # Configurer le style des graphiques
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Créer un dossier pour sauvegarder les graphiques si nécessaire
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        # --- Graphique 1: Evolution du portefeuille ---
        fig, ax1 = plt.subplots()
        
        # Tracer l'évolution du capital
        ax1.plot(self.results['timestamps'], self.results['portfolio_values'], 
                label="Valeur totale", color='blue', linewidth=2)
        
        # Ajouter une ligne pour la stratégie buy & hold
        buy_hold_values = self.initial_balance * self.results['prices'] / self.results['prices'][0]
        ax1.plot(self.results['timestamps'], buy_hold_values, 
                label="Buy & Hold", color='green', linestyle='--', alpha=0.7)
        
        # Configurer l'axe Y
        ax1.set_ylabel('Valeur ($)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Configurer l'axe X
        ax1.set_xlabel('Date')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Ajouter un deuxième axe Y pour le prix
        ax2 = ax1.twinx()
        ax2.plot(self.results['timestamps'], self.results['prices'], 
                label="Prix", color='red', alpha=0.5)
        ax2.set_ylabel('Prix ($)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Ajouter les trades sur le graphique
        for trade in self.results['trades']:
            idx = trade['step'] - self.window_size
            if idx >= 0 and idx < len(self.results['timestamps']):
                timestamp = self.results['timestamps'][idx]
                price = self.results['prices'][idx]
                
                if trade['type'] == 'buy':
                    ax1.scatter(timestamp, self.results['portfolio_values'][idx], 
                               color='green', marker='^', s=100, zorder=5)
                else:  # sell
                    ax1.scatter(timestamp, self.results['portfolio_values'][idx], 
                               color='red', marker='v', s=100, zorder=5)
        
        # Ajouter un titre et une légende
        plt.title('Évolution du portefeuille vs Buy & Hold', fontsize=16)
        fig.tight_layout()
        
        # Combiner les légendes des deux axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'portfolio_evolution.png'), dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # --- Graphique 2: Allocation des actifs ---
        fig, ax = plt.subplots()
        
        # Calculer la valeur de la crypto en dollars
        crypto_values = [c * p for c, p in zip(self.results['crypto_holdings'], self.results['prices'])]
        
        # Créer un graphique empilé
        ax.stackplot(self.results['timestamps'], 
                    [self.results['balances'], crypto_values], 
                    labels=['USD', 'Crypto'],
                    colors=['#1f77b4', '#ff7f0e'], alpha=0.7)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Allocation ($)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.title('Allocation des actifs au fil du temps', fontsize=16)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'asset_allocation.png'), dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # --- Graphique 3: Actions et récompenses ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Tracer les actions
        ax1.plot(self.results['timestamps'], self.results['actions'], color='purple')
        ax1.set_ylabel('Action [-1, 1]')
        ax1.set_title('Actions prises par l\'agent', fontsize=14)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax1.fill_between(self.results['timestamps'], self.results['actions'], 0, 
                        where=[a > 0 for a in self.results['actions']], 
                        color='green', alpha=0.3, interpolate=True)
        ax1.fill_between(self.results['timestamps'], self.results['actions'], 0, 
                        where=[a < 0 for a in self.results['actions']], 
                        color='red', alpha=0.3, interpolate=True)
        
        # Tracer les récompenses
        ax2.plot(self.results['timestamps'], self.results['rewards'], color='orange')
        ax2.set_ylabel('Récompense')
        ax2.set_xlabel('Date')
        ax2.set_title('Récompenses obtenues', fontsize=14)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'actions_rewards.png'), dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # --- Graphique 4: Métriques de performance ---
        metrics_to_plot = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        metrics_values = [self.results[m] for m in metrics_to_plot]
        metrics_labels = ['Rendement Total (%)', 'Ratio de Sharpe', 'Drawdown Max (%)', 'Taux de Réussite (%)']
        
        fig, ax = plt.subplots()
        bars = ax.bar(metrics_labels, metrics_values, color=['blue', 'green', 'red', 'orange'])
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.title('Métriques de Performance', fontsize=16)
        plt.ylabel('Valeur')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'performance_metrics.png'), dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Affichage des principales métriques
        print("\n--- Résumé des performances ---")
        print(f"Rendement total: {self.results['total_return']:.2f}%")
        print(f"Rendement annualisé: {self.results['annualized_return']:.2f}%")
        print(f"Volatilité annualisée: {self.results['annualized_volatility']:.2f}%")
        print(f"Ratio de Sharpe: {self.results['sharpe_ratio']:.2f}")
        print(f"Drawdown Maximum: {self.results['max_drawdown']:.2f}%")
        print(f"Nombre de trades: {self.results['n_trades']}")
        print(f"Taux de réussite: {self.results['win_rate']:.2f}%")
        print(f"Rendement moyen par trade: {self.results['avg_trade_return']:.2f}%")
        print("-----------------------------")


if __name__ == "__main__":
    # Exemple d'utilisation
    from data_loader import CryptoDataLoader
    
    # Charger les données
    loader = CryptoDataLoader()
    df = loader.load_or_download("ETHUSDT", interval="1h", days_back=60)
    
    # Diviser en train/test
    train_df, test_df = loader.prepare_data_for_training(df)
    
    # Créer l'environnement de backtest
    backtester = Backtester(test_df)
    
    # Charger un modèle entraîné
    agent = create_agent("ppo", backtester.env)
    agent.load("models/ppo_trader")
    
    # Exécuter le backtest
    results = backtester.run_backtest(agent)
    
    # Afficher les résultats
    backtester.plot_results(save_path="results")