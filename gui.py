import sys
import threading
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QTextEdit, QComboBox, QSpinBox, QFileDialog
)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Modules IA
from V2.data_loader import DataLoader
from V2.env import TradingEnv
from V2.agents import TradingCallback
from stable_baselines3 import PPO, DQN, A2C

class AIInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interface IA Trading")
        self.resize(800, 600)
        self.layout = QVBoxLayout()
        # Sélecteur modèle
        self.layout.addWidget(QLabel("Modèle:"))
        self.model_box = QComboBox()
        self.model_box.addItems(["PPO", "DQN", "A2C"])
        self.layout.addWidget(self.model_box)
        # Hyperparamètres epochs
        self.layout.addWidget(QLabel("Nombre d'épisodes:"))
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 10000)
        self.episodes_spin.setValue(100)
        self.layout.addWidget(self.episodes_spin)
        # Charger données
        self.load_btn = QPushButton("Charger données")
        self.load_btn.clicked.connect(self.load_data)
        self.layout.addWidget(self.load_btn)
        # Bouton entraînement
        self.train_btn = QPushButton("Entraîner")
        self.train_btn.clicked.connect(self.train_model)
        self.layout.addWidget(self.train_btn)
        # Bouton évaluation
        self.eval_btn = QPushButton("Évaluer")
        self.eval_btn.clicked.connect(self.evaluate_model)
        self.layout.addWidget(self.eval_btn)
        # Log
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addWidget(self.log_area)
        # Graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        # Variables
        self.env = None
        self.model = None
        self.data = None

    def log(self, msg):
        self.log_area.append(msg)
        self.log_area.repaint()

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Sélectionner fichier CSV", "", "CSV Files (*.csv)")
        if path:
            self.log(f"Chargement: {path}")
            self.data = DataLoader(path).load()
            self.env = TradingEnv(self.data)
            self.log("Données chargées, env ok.")

    def train_thread(self):
        algo = self.model_box.currentText()
        episodes = self.episodes_spin.value()
        self.log(f"Start entraînement {algo}, {episodes} épisodes.")
        # Sélection
        if algo == "PPO":
            self.model = PPO("MlpPolicy", self.env, verbose=0)
        elif algo == "DQN":
            self.model = DQN("MlpPolicy", self.env, verbose=0)
        else:
            self.model = A2C("MlpPolicy", self.env, verbose=0)
        # Callback
        callback = TradingCallback()
        # Entraînement
        self.model.learn(total_timesteps=episodes * len(self.env), callback=callback)
        self.log("Entraînement terminé.")

    def train_model(self):
        if not self.env:
            self.log("Erreur: données non chargées.")
            return
        thread = threading.Thread(target=self.train_thread)
        thread.start()

    def evaluate_model(self):
        if not self.model:
            self.log("Erreur: modèle non entraîné.")
            return
        self.log("Évaluation...")
        mean_reward, std_reward = self.model.evaluate_policy(self.model, self.env, n_eval_episodes=10)
        self.log(f"Résultat: {mean_reward:.2f}±{std_reward:.2f}")
        # Affichage graphique
        rewards = [mean_reward] * 10  # placeholder
        self.ax.clear()
        self.ax.plot(rewards)
        self.ax.set_title("Récompenses évaluation")
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIInterface()
    window.show()
    sys.exit(app.exec())
