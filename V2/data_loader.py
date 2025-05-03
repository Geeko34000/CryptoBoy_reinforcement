import numpy as np
import os
import torch as th
from typing import Dict, Any, Union, Type, Optional
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import Env


class TradingCallback(BaseCallback):
    """
    Callback pour surveiller et enregistrer les modèles pendant l'entraînement.
    """
    
    def __init__(self, save_path: str = "models", save_freq: int = 10000, 
                 model_name: str = "model", verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.model_name = model_name
        
        # Créer le dossier de sauvegarde si nécessaire
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    def _init_callback(self) -> None:
        # Créer le chemin de sauvegarde
        self.save_path = os.path.join(self.save_path, self.model_name)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}_{self.n_calls}_steps"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Modèle sauvegardé à {model_path}")
        return True


class TradingAgent:
    """
    Classe abstraite pour tous les agents de trading.
    """
    
    def __init__(self, env: Union[Env, VecEnv], model_name: str = "trading_agent"):
        self.env = env
        self.model_name = model_name
        self.model = None
        
    def train(self, total_timesteps: int, callback: Optional[Union[BaseCallback, list]] = None) -> None:
        """Méthode d'entraînement à implémenter par les sous-classes."""
        raise NotImplementedError
        
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Prédit la meilleure action pour une observation donnée."""
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entrainé ou chargé.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, path: str) -> None:
        """Sauvegarde le modèle."""
        if self.model is None:
            raise ValueError("Pas de modèle à sauvegarder.")
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """Charge un modèle entraîné."""
        raise NotImplementedError
    
    def evaluate(self, n_eval_episodes: int = 10) -> Dict[str, float]:
        """Évalue le modèle sur plusieurs épisodes."""
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entrainé ou chargé.")
        mean_reward, std_reward = evaluate_policy(
            self.model, self.env, n_eval_episodes=n_eval_episodes
        )
        return {"mean_reward": mean_reward, "std_reward": std_reward}


class PPOAgent(TradingAgent):
    """
    Agent utilisant l'algorithme Proximal Policy Optimization (PPO).
    """
    
    def __init__(self, env: Union[Env, VecEnv], model_name: str = "ppo_trader",
                 policy: str = "MlpPolicy", **kwargs):
        super().__init__(env, model_name)
        self.policy = policy
        self.kwargs = kwargs
        
        # Paramètres par défaut optimisés pour le trading
        default_params = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "verbose": 1,
            "tensorboard_log": "./logs/"
        }
        
        # Mettre à jour avec les paramètres fournis
        self.params = {**default_params, **kwargs}
        
        # Initialiser le modèle
        self.model = PPO(
            policy=self.policy,
            env=self.env,
            **self.params
        )
    
    def train(self, total_timesteps: int, callback: Optional[Union[BaseCallback, list]] = None) -> None:
        """Entraîne l'agent PPO."""
        # Créer le callback de sauvegarde par défaut
        save_callback = TradingCallback(
            save_path="models",
            save_freq=max(10000, total_timesteps // 10),
            model_name=self.model_name
        )
        
        # Combiner avec d'autres callbacks si nécessaire
        if callback:
            if isinstance(callback, list):
                callbacks = CallbackList([save_callback] + callback)
            else:
                callbacks = CallbackList([save_callback, callback])
        else:
            callbacks = save_callback
        
        # Entraîner le modèle
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
    
    def load(self, path: str) -> None:
        """Charge un modèle PPO entraîné."""
        self.model = PPO.load(path, env=self.env)


class DQNAgent(TradingAgent):
    """
    Agent utilisant l'algorithme Deep Q-Network (DQN).
    """
    
    def __init__(self, env: Union[Env, VecEnv], model_name: str = "dqn_trader",
                 policy: str = "MlpPolicy", **kwargs):
        super().__init__(env, model_name)
        self.policy = policy
        self.kwargs = kwargs
        
        # Paramètres par défaut optimisés pour le trading
        default_params = {
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 64,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.2,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "verbose": 1,
            "tensorboard_log": "./logs/"
        }
        
        # Mettre à jour avec les paramètres fournis
        self.params = {**default_params, **kwargs}
        
        # Initialiser le modèle
        self.model = DQN(
            policy=self.policy,
            env=self.env,
            **self.params
        )
    
    def train(self, total_timesteps: int, callback: Optional[Union[BaseCallback, list]] = None) -> None:
        """Entraîne l'agent DQN."""
        # Créer le callback de sauvegarde par défaut
        save_callback = TradingCallback(
            save_path="models",
            save_freq=max(10000, total_timesteps // 10),
            model_name=self.model_name
        )
        
        # Combiner avec d'autres callbacks si nécessaire
        if callback:
            if isinstance(callback, list):
                callbacks = CallbackList([save_callback] + callback)
            else:
                callbacks = CallbackList([save_callback, callback])
        else:
            callbacks = save_callback
        
        # Entraîner le modèle
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100
        )
    
    def load(self, path: str) -> None:
        """Charge un modèle DQN entraîné."""
        self.model = DQN.load(path, env=self.env)


class A2CAgent(TradingAgent):
    """
    Agent utilisant l'algorithme Advantage Actor-Critic (A2C).
    """
    
    def __init__(self, env: Union[Env, VecEnv], model_name: str = "a2c_trader",
                 policy: str = "MlpPolicy", **kwargs):
        super().__init__(env, model_name)
        self.policy = policy
        self.kwargs = kwargs
        
        # Paramètres par défaut optimisés pour le trading
        default_params = {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "rms_prop_eps": 1e-5,
            "verbose": 1,
            "tensorboard_log": "./logs/"
        }
        
        # Mettre à jour avec les paramètres fournis
        self.params = {**default_params, **kwargs}
        
        # Initialiser le modèle
        self.model = A2C(
            policy=self.policy,
            env=self.env,
            **self.params
        )
    
    def train(self, total_timesteps: int, callback: Optional[Union[BaseCallback, list]] = None) -> None:
        """Entraîne l'agent A2C."""
        # Créer le callback de sauvegarde par défaut
        save_callback = TradingCallback(
            save_path="models",
            save_freq=max(10000, total_timesteps // 10),
            model_name=self.model_name
        )
        
        # Combiner avec d'autres callbacks si nécessaire
        if callback:
            if isinstance(callback, list):
                callbacks = CallbackList([save_callback] + callback)
            else:
                callbacks = CallbackList([save_callback, callback])
        else:
            callbacks = save_callback
        
        # Entraîner le modèle
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
    
    def load(self, path: str) -> None:
        """Charge un modèle A2C entraîné."""
        self.model = A2C.load(path, env=self.env)


def create_agent(agent_type: str, env: Union[Env, VecEnv], **kwargs) -> TradingAgent:
    """
    Crée un agent de trading selon le type spécifié.
    
    Args:
        agent_type: Type d'agent à créer ('ppo', 'dqn', 'a2c')
        env: Environnement de trading
        **kwargs: Paramètres supplémentaires pour l'agent
    
    Returns:
        Un objet TradingAgent
    """
    agent_types = {
        'ppo': PPOAgent,
        'dqn': DQNAgent,
        'a2c': A2CAgent
    }
    
    if agent_type not in agent_types:
        raise ValueError(f"Type d'agent inconnu: {agent_type}. Options valides: {list(agent_types.keys())}")
    
    return agent_types[agent_type](env=env, **kwargs)