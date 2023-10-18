from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque

import numpy as np
import random as random

class DQNAgent:

    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 initial_epsilon: float,
                 final_epsilon: float,
                 epsilon_decay: float,
                 gamma: float,
                 learning_rate: float) -> None:
        """
        Initialize the agent.
        """
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.memory = deque(maxlen=2000)

        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential()

        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model
    
    def remember(self, state: np.array, action: int, rewards: int, next_state: np.array, done: bool) -> None:
        self.memory.append((state, action, rewards, next_state, done))

    def act(self, state: np.array) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.model.predict(state, verbose=0)[0])
        
    def replay(self, batch_size: int) -> None:
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward # If done, then target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target # Update the Q value for the prediction in the Q table

            self.model.fit(state, target_f, epochs=1, verbose=0)

        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def load(self, name: str) -> None:
        self.model.load_weights(name)

    def save(self, name: str) -> None:
        self.model.save_weights(name)