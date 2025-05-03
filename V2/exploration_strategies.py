import numpy as np
import torch as th
from typing import Dict, Any, Union, Callable, List, Optional, Tuple
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise, OrnsteinUhlenbeckActionNoise


class AdaptiveNoiseStrategy:
    """
    Stratégie de bruit adaptative qui ajuste l'intensité du bruit 
    en fonction des performances récentes.
    """
    
    def __init__(self, initial_std: float = 0.3, 
                 min_std: float = 0.05, 
                 max_std: float = 0.5,
                 adaptation_coef: float = 0.05,
                 history_length: int = 50):
        """
        Initialise la stratégie de bruit adaptative.
        
        Args:
            initial_std: Écart-type initial du bruit
            min_std: Écart-type minimum du bruit
            max_std: Écart-type maximum du bruit
            adaptation_coef: Coefficient d'adaptation du bruit
            history_length: Longueur de l'historique des récompenses à considérer
        """
        self.current_std = initial_std
        self.min_std = min_std
        self.max_std = max_std
        self.adaptation_coef = adaptation_coef
        self.history_length = history_length
        self.reward_history = []
        self.noise_type = "normal"  # Par défaut
    
    def create_action_noise(self, action_dim: int, noise_type: str = "normal") -> ActionNoise:
        """
        Crée un objet de bruit d'action selon le type spécifié.
        
        Args:
            action_dim: Dimension de l'espace d'action
            noise_type: Type de bruit ('normal' ou 'ou' pour Ornstein-Uhlenbeck)
            
        Returns:
            Un objet ActionNoise
        """
        self.noise_type = noise_type
        
        if noise_type == "normal":
            return NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=self.current_std * np.ones(action_dim)
            )
        elif noise_type == "ou":
            return OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(action_dim),
                sigma=self.current_std * np.ones(action_dim),
                theta=0.15,
                dt=1e-2
            )
        else:
            raise ValueError(f"Type de bruit inconnu: {noise_type}. Options valides: 'normal', 'ou'")
    
    def update_noise_parameters(self, reward: float) -> ActionNoise:
        """
        Met à jour les paramètres du bruit en fonction de la récompense reçue.
        
        Args:
            reward: Récompense obtenue dans l'épisode actuel
            
        Returns:
            Un nouvel objet ActionNoise avec les paramètres mis à jour
        """
        # Ajouter la récompense à l'historique
        self.reward_history.append(reward)
        
        # Limiter la taille de l'historique
        if len(self.reward_history) > self.history_length:
            self.reward_history = self.reward_history[-self.history_length:]
        
        # Ne pas ajuster le bruit si l'historique est trop court
        if len(self.reward_history) < 5:
            return None
        
        # Calculer la tendance des récompenses récentes
        recent_rewards = self.reward_history[-10:]
        if len(recent_rewards) >= 2:
            # Si les récompenses récentes augmentent, réduire le bruit
            if np.mean(recent_rewards[-5:]) > np.mean(recent_rewards[:-5]):
                self.current_std *= (1 - self.adaptation_coef)
            # Si les récompenses stagnent ou diminuent, augmenter le bruit
            else:
                self.current_std *= (1 + self.adaptation_coef)
            
            # Limiter l'écart-type dans les bornes définies
            self.current_std = np.clip(self.current_std, self.min_std, self.max_std)
        
        # Créer et retourner un nouvel objet de bruit avec les paramètres mis à jour
        action_dim = 1  # Pour l'environnement de trading standard
        return self.create_action_noise(action_dim, self.noise_type)


class CurriculumLearningStrategy:
    """
    Stratégie d'apprentissage par curriculum qui augmente progressivement
    la difficulté de l'environnement d'entraînement.
    """
    
    def __init__(self, initial_window: int = 20, 
                 max_window: int = 100, 
                 window_increment: int = 5,
                 performance_threshold: float = 0.05,
                 min_episodes_per_level: int = 20):
        """
        Initialise la stratégie d'apprentissage par curriculum.
        
        Args:
            initial_window: Taille initiale de la fenêtre d'observation
            max_window: Taille maximale de la fenêtre d'observation
            window_increment: Incrément de la taille de la fenêtre
            performance_threshold: Seuil de performance pour passer au niveau suivant
            min_episodes_per_level: Nombre minimum d'épisodes par niveau
        """
        self.current_window = initial_window
        self.max_window = max_window
        self.window_increment = window_increment
        self.performance_threshold = performance_threshold
        self.min_episodes_per_level = min_episodes_per_level
        
        self.episode_count = 0
        self.level_reward_history = []
    
    def update_difficulty(self, mean_reward: float) -> Tuple[bool, int]:
        """
        Met à jour la difficulté de l'environnement en fonction des performances.
        
        Args:
            mean_reward: Récompense moyenne des derniers épisodes
            
        Returns:
            Tuple (need_update, new_window) indiquant s'il faut mettre à jour 
            l'environnement et la nouvelle taille de fenêtre
        """
        self.episode_count += 1
        self.level_reward_history.append(mean_reward)
        
        # Limiter la taille de l'historique des récompenses pour ce niveau
        if len(self.level_reward_history) > 20:
            self.level_reward_history = self.level_reward_history[-20:]
        
        # Vérifier si on peut passer au niveau suivant
        if (self.episode_count >= self.min_episodes_per_level and 
            len(self.level_reward_history) >= 10):
            
            avg_reward = np.mean(self.level_reward_history[-10:])
            
            # Si la performance est suffisante et on n'a pas atteint la difficulté maximale
            if avg_reward > self.performance_threshold and self.current_window < self.max_window:
                self.current_window = min(self.current_window + self.window_increment, self.max_window)
                self.episode_count = 0
                self.level_reward_history = []
                return True, self.current_window
        
        return False, self.current_window


class ExplorationManager:
    """
    Gestionnaire d'exploration qui coordonne différentes stratégies d'exploration.
    """
    
    def __init__(self):
        """Initialise le gestionnaire d'exploration."""
        self.noise_strategy = None
        self.curriculum_strategy = None
        self.episodic_rewards = []
    
    def initialize_noise_strategy(self, action_dim: int = 1, 
                                 noise_type: str = "normal",
                                 **kwargs) -> ActionNoise:
        """
        Initialise la stratégie de bruit.
        
        Args:
            action_dim: Dimension de l'espace d'action
            noise_type: Type de bruit ('normal' ou 'ou')
            **kwargs: Paramètres supplémentaires pour la stratégie de bruit
            
        Returns:
            Un objet ActionNoise initial
        """
        # Paramètres par défaut
        default_params = {
            "initial_std": 0.3,
            "min_std": 0.05,
            "max_std": 0.5,
            "adaptation_coef": 0.05,
            "history_length": 50
        }
        
        # Mettre à jour avec les paramètres fournis
        params = {**default_params, **kwargs}
        
        self.noise_strategy = AdaptiveNoiseStrategy(
            initial_std=params["initial_std"],
            min_std=params["min_std"],
            max_std=params["max_std"],
            adaptation_coef=params["adaptation_coef"],
            history_length=params["history_length"]
        )
        
        return self.noise_strategy.create_action_noise(action_dim, noise_type)
    
    def initialize_curriculum_strategy(self, **kwargs) -> None:
        """
        Initialise la stratégie d'apprentissage par curriculum.
        
        Args:
            **kwargs: Paramètres pour la stratégie de curriculum
        """
        # Paramètres par défaut
        default_params = {
            "initial_window": 20,
            "max_window": 100,
            "window_increment": 5,
            "performance_threshold": 0.05,
            "min_episodes_per_level": 20
        }
        
        # Mettre à jour avec les paramètres fournis
        params = {**default_params, **kwargs}
        
        self.curriculum_strategy = CurriculumLearningStrategy(
            initial_window=params["initial_window"],
            max_window=params["max_window"],
            window_increment=params["window_increment"],
            performance_threshold=params["performance_threshold"],
            min_episodes_per_level=params["min_episodes_per_level"]
        )
    
    def register_episode_reward(self, reward: float) -> Tuple[ActionNoise, Tuple[bool, int]]:
        """
        Enregistre la récompense d'un épisode et met à jour les stratégies.
        
        Args:
            reward: Récompense totale de l'épisode
            
        Returns:
            Tuple (noise, curriculum_update) où noise est le nouvel objet de bruit
            et curriculum_update est un tuple (need_update, new_window)
        """
        self.episodic_rewards.append(reward)
        
        # Limiter la taille de l'historique
        if len(self.episodic_rewards) > 100:
            self.episodic_rewards = self.episodic_rewards[-100:]
        
        # Calculer la récompense moyenne récente
        mean_reward = np.mean(self.episodic_rewards[-20:]) if len(self.episodic_rewards) >= 20 else np.mean(self.episodic_rewards)
        
        # Mettre à jour la stratégie de bruit si elle existe
        noise = None
        if self.noise_strategy is not None:
            noise = self.noise_strategy.update_noise_parameters(mean_reward)
        
        # Mettre à jour la stratégie de curriculum si elle existe
        curriculum_update = (False, 0)
        if self.curriculum_strategy is not None:
            curriculum_update = self.curriculum_strategy.update_difficulty(mean_reward)
        
        return noise, curriculum_update


class ExplorationCallback:
    """
    Callback pour gérer l'exploration pendant l'entraînement.
    """
    
    def __init__(self, exploration_manager: ExplorationManager, 
                env_update_fn: Callable = None):
        """
        Initialise le callback d'exploration.
        
        Args:
            exploration_manager: Le gestionnaire d'exploration
            env_update_fn: Fonction pour mettre à jour l'environnement
        """
        self.exploration_manager = exploration_manager
        self.env_update_fn = env_update_fn
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_step = 0
        self.episodes_completed = 0
    
    def on_step(self, locals_dict: Dict, globals_dict: Dict) -> None:
        """
        Callback appelé à chaque étape de l'environnement.
        
        Args:
            locals_dict: Variables locales du contexte d'exécution
            globals_dict: Variables globales du contexte d'exécution
        """
        # Extraire les informations nécessaires
        rewards = locals_dict.get('rewards')
        dones = locals_dict.get('dones')
        infos = locals_dict.get('infos')
        
        if rewards is None or dones is None:
            return
        
        # Pour les environnements vectorisés
        if isinstance(rewards, (list, np.ndarray)):
            for i, (reward, done) in enumerate(zip(rewards, dones)):
                self.current_episode_reward += reward
                self.episode_step += 1
                
                if done:
                    # Enregistrer la récompense de l'épisode terminé
                    self.episode_rewards.append(self.current_episode_reward)
                    self.episodes_completed += 1
                    
                    # Mettre à jour les stratégies d'exploration
                    new_noise, (update_env, new_window) = self.exploration_manager.register_episode_reward(
                        self.current_episode_reward
                    )
                    
                    # Mettre à jour le bruit si nécessaire
                    model = locals_dict.get('self')
                    if new_noise is not None and hasattr(model, 'action_noise'):
                        model.action_noise = new_noise
                    
                    # Mettre à jour l'environnement si nécessaire
                    if update_env and self.env_update_fn is not None:
                        self.env_update_fn(new_window)
                    
                    # Réinitialiser pour le prochain épisode
                    self.current_episode_reward = 0
                    self.episode_step = 0
        else:
            # Pour les environnements non vectorisés
            self.current_episode_reward += rewards
            self.episode_step += 1
            
            if dones:
                # Même logique que ci-dessus
                self.episode_rewards.append(self.current_episode_reward)
                self.episodes_completed += 1
                
                new_noise, (update_env, new_window) = self.exploration_manager.register_episode_reward(
                    self.current_episode_reward
                )
                
                model = locals_dict.get('self')
                if new_noise is not None and hasattr(model, 'action_noise'):
                    model.action_noise = new_noise
                
                if update_env and self.env_update_fn is not None:
                    self.env_update_fn(new_window)
                
                self.current_episode_reward = 0
                self.episode_step = 0