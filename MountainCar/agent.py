from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque

import numpy as np
import random as random

class ContinuousAgent:

    def __init__(self, 
                 state_size: int,
                 initial_epsilon: float,
                 final_epsilon: float,
                 epsilon_decay: float,
                 gamma: float,
                 learning_rate: float) -> None:
        """
        Initialize the agent.
        """
        self.state_size = state_size

        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.episode = 0

        self.gamma = gamma
        self.learning_rate = learning_rate

    def act(self, state: np.array) -> int:
        pass

    def load(self, name: str) -> None:
        pass

    def save(self, name: str) -> None:
        pass