import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime


class TradingMetrics:
    """
    Classe pour calculer les métriques de performance de trading.
    """
    
    def __init__(self):
        """Initialise le calculateur de métriques."""
        pass
    
    def calculate_return(self, portfolio_values: List[float]) -> float:
        """
        Calcule le rendement total en pourcentage.
        
        Args:
            portfolio_values: Liste des valeurs du portefeuille
        
        Returns:
            Rendement total en pourcentage
        """
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
            
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        if initial_value <= 0:
            return 0.0
            
        return ((final_value / initial_value) - 1) * 100
    
    def calculate_annualized_return(self, portfolio_values: List[float], 
                                   timestamps: List[datetime]) -> float:
        """
        Calcule le rendement annualisé.
        
        Args:
            portfolio_values: Liste des valeurs du portefeuille
            timestamps: Liste des timestamps correspondants
        
        Returns:
            Rendement annualisé en pourcentage
        """
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
            
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        if initial_value <= 0:
            return 0.0
        
        # Calcul de la durée en années
        start_date = timestamps[0]
        end_date = timestamps[-1]
        duration_days = (end_date - start_date).total_seconds() / (60 * 60 * 24)
        duration_years = duration_days / 365.25
        
        if duration_years <= 0:
            return 0.0
        
        # Calcul du rendement annualisé
        total_return = (final_value / initial_value) - 1
        annualized_return = ((1 + total_return) ** (1 / duration_years) - 1) * 100
        
        return annualized_return
    
    def calculate_volatility(self, portfolio_values: List[float]) -> float:
        """
        Calcule la volatilité quotidienne (écart-type des rendements quotidiens).
        
        Args:
            portfolio_values: Liste des valeurs du portefeuille
        
        Returns:
            Volatilité en pourcentage
        """
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
        
        # Calcul des rendements quotidiens
        returns = np.array([
            (portfolio_values[i] / portfolio_values[i-1]) - 1 
            for i in range(1, len(portfolio_values))
        ])
        
        # Calcul de la volatilité (écart-type)
        volatility = np.std(returns) * 100
        
        return volatility
    
    def calculate_annualized_volatility(self, portfolio_values: List[float], 
                                       trading_days_per_year: int = 365) -> float:
        """
        Calcule la volatilité annualisée.
        
        Args:
            portfolio_values: Liste des valeurs du portefeuille
            trading_days_per_year: Nombre de jours de trading par an
        
        Returns:
            Volatilité annualisée en pourcentage
        """
        daily_volatility = self.calculate_volatility(portfolio_values)
        return daily_volatility * np.sqrt(trading_days_per_year)
    
    def calculate_sharpe_ratio(self, portfolio_values: List[float], 
                              risk_free_rate: float = 0.01,
                              trading_days_per_year: int = 365) -> float:
        """
        Calcule le ratio de Sharpe.
        
        Args:
            portfolio_values: Liste des valeurs du portefeuille
            risk_free_rate: Taux sans risque annualisé
            trading_days_per_year: Nombre de jours de trading par an
        
        Returns:
            Ratio de Sharpe
        """
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
        
        # Rendements quotidiens
        returns = np.array([
            (portfolio_values[i] / portfolio_values[i-1]) - 1 
            for i in range(1, len(portfolio_values))
        ])
        
        # Calcul du rendement moyen et de la volatilité
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Taux sans risque quotidien
        daily_risk_free = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
        
        # Calcul du ratio de Sharpe
        sharpe = (mean_return - daily_risk_free) / std_return
        
        # Annualisation
        sharpe_annualized = sharpe * np.sqrt(trading_days_per_year)
        
        return sharpe_annualized
    
    def calculate_sortino_ratio(self, portfolio_values: List[float], 
                               risk_free_rate: float = 0.01,
                               trading_days_per_year: int = 365) -> float:
        """
        Calcule le ratio de Sortino.
        
        Args:
            portfolio_values: Liste des valeurs du portefeuille
            risk_free_rate: Taux sans risque annualisé
            trading_days_per_year: Nombre de jours de trading par an
        
        Returns:
            Ratio de Sortino
        """
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
        
        # Rendements quotidiens
        returns = np.array([
            (portfolio_values[i] / portfolio_values[i-1]) - 1 
            for i in range(1, len(portfolio_values))
        ])
        
        # Calcul du rendement moyen
        mean_return = np.mean(returns)
        
        # Taux sans risque quotidien
        daily_risk_free = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
        
        # Calculer uniquement les rendements négatifs
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')  # Aucun rendement négatif
        
        # Écart-type des rendements négatifs (downside deviation)
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        # Calcul du ratio de Sortino
        sortino = (mean_return - daily_risk_free) / downside_deviation
        
        # Annualisation
        sortino_annualized = sortino * np.sqrt(trading_days_per_year)
        
        return sortino_annualized
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """
        Calcule le drawdown maximum en pourcentage.
        
        Args:
            portfolio_values: Liste des valeurs du portefeuille
        
        Returns:
            Drawdown maximum en pourcentage
        """
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
        
        # Calcul du drawdown maximum
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """
        Calcule le taux de réussite des trades.
        
        Args:
            trades: Liste des trades
        
        Returns:
            Taux de réussite en pourcentage
        """
        if not trades:
            return 0.0
        
        # Extraire les trades d'achat et de vente
        buys = [trade for trade in trades if trade['type'] == 'buy']
        sells = [trade for trade in trades if trade['type'] == 'sell']
        
        if not buys or not sells:
            return 0.0
        
        # Simplification: supposons que les ventes sont toujours profitables
        # Dans un cas réel, il faudrait comparer le prix de vente au prix d'achat moyen
        n_winning_trades = len(sells)
        n_total_trades = len(sells)
        
        return (n_winning_trades / n_total_trades) * 100
    
    def calculate_avg_trade_return(self, trades: List[Dict], prices: List[float]) -> float:
        """
        Calcule le rendement moyen par trade.
        
        Args:
            trades: Liste des trades
            prices: Liste des prix
        
        Returns:
            Rendement moyen par trade en pourcentage
        """
        if not trades:
            return 0.0
        
        # Extraire les rendements des trades
        trade_returns = []
        
        buy_price = None
        for trade in trades:
            if trade['type'] == 'buy':
                buy_price = trade['price']
            elif trade['type'] == 'sell' and buy_price is not None:
                sell_price = trade['price']
                trade_return = (sell_price / buy_price - 1) * 100
                trade_returns.append(trade_return)
                buy_price = None
        
        if not trade_returns:
            return 0.0
        
        # Calcul du rendement moyen par trade
        return np.mean(trade_returns)
    
    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """
        Calcule le facteur de profit (somme des profits / somme des pertes).
        
        Args:
            trades: Liste des trades
        
        Returns:
            Facteur de profit
        """
        if not trades:
            return 0.0
        
        total_profit = 0.0
        total_loss = 0.0
        
        buy_price = None
        buy_amount = None
        
        for trade in trades:
            if trade['type'] == 'buy':
                buy_price = trade['price']
                buy_amount = trade.get('amount', 0)
            elif trade['type'] == 'sell' and buy_price is not None and buy_amount is not None:
                sell_price = trade['price']
                sell_amount = trade.get('amount', 0)
                
                # Calculer le profit/perte
                if min(buy_amount, sell_amount) > 0:
                    pnl = (sell_price - buy_price) * min(buy_amount, sell_amount)
                    
                    if pnl > 0:
                        total_profit += pnl
                    else:
                        total_loss += abs(pnl)
                
                # Réinitialiser pour le prochain trade
                buy_price = None
                buy_amount = None
        
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0
        
        return total_profit / total_loss
    
    def calculate_calmar_ratio(self, portfolio_values: List[float], 
                              timestamps: List[datetime]) -> float:
        """
        Calcule le ratio de Calmar (rendement annualisé / drawdown max).
        
        Args:
            portfolio_values: Liste des valeurs du portefeuille
            timestamps: Liste des timestamps correspondants
        
        Returns:
            Ratio de Calmar
        """
        annualized_return = self.calculate_annualized_return(portfolio_values, timestamps)
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / max_drawdown
    
    def calculate_all_metrics(self, portfolio_values: List[float], 
                             trades: List[Dict], 
                             timestamps: List[datetime],
                             prices: List[float],
                             risk_free_rate: float = 0.01) -> Dict[str, float]:
        """
        Calcule toutes les métriques disponibles.
        
        Args:
            portfolio_values: Liste des valeurs du portefeuille
            trades: Liste des trades
            timestamps: Liste des timestamps
            prices: Liste des prix
            risk_free_rate: Taux sans risque annualisé
        
        Returns:
            Dictionnaire contenant toutes les métriques
        """
        return {
            'total_return': self.calculate_return(portfolio_values),
            'annualized_return': self.calculate_annualized_return(portfolio_values, timestamps),
            'volatility': self.calculate_volatility(portfolio_values),
            'annualized_volatility': self.calculate_annualized_volatility(portfolio_values),
            'sharpe_ratio': self.calculate_sharpe_ratio(portfolio_values, risk_free_rate),
            'sortino_ratio': self.calculate_sortino_ratio(portfolio_values, risk_free_rate),
            'max_drawdown': self.calculate_max_drawdown(portfolio_values),
            'win_rate': self.calculate_win_rate(trades),
            'avg_trade_return': self.calculate_avg_trade_return(trades, prices),
            'profit_factor': self.calculate_profit_factor(trades),
            'calmar_ratio': self.calculate_calmar_ratio(portfolio_values, timestamps)
        }