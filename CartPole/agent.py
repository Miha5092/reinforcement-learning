from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque

import numpy as np
import random as random

import time

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

        self.memory_size = 5000

        self.state_memory = deque(maxlen=self.memory_size)
        self.action_memory = deque(maxlen=self.memory_size)
        self.reward_memory = deque(maxlen=self.memory_size)
        self.next_state_memory = deque(maxlen=self.memory_size)
        self.done_memory = deque(maxlen=self.memory_size)

        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential()

        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model
    
    def remember(self, state: np.array, action: int, rewards: int, next_state: np.array, done: bool) -> None:
        # self.memory.append([state, action, rewards, next_state, done])
        # print(state[0], action, rewards, next_state[0], done)

        self.state_memory.append(state[0])
        self.action_memory.append(action)
        self.reward_memory.append(rewards)
        self.next_state_memory.append(next_state[0])
        self.done_memory.append(done)

    def act(self, state: np.array) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.model.predict(state, verbose=0)[0])
        
    def replay(self, batch_size: int) -> None:
        minibatch_indices = np.random.choice(len(self.state_memory), batch_size, replace=False)

        state_minibatch = np.array(self.state_memory)[minibatch_indices]
        action_minibatch = np.array(self.action_memory)[minibatch_indices]
        reward_minibatch = np.array(self.reward_memory)[minibatch_indices]
        next_state_minibatch = np.array(self.next_state_memory)[minibatch_indices]
        done_minibatch = np.array(self.done_memory)[minibatch_indices]

        temporal_difference = np.amax(self.model.predict(next_state_minibatch, verbose=0), axis=1) * self.gamma * (1 - done_minibatch)
        target = reward_minibatch + temporal_difference

        target_f = self.model.predict(state_minibatch, verbose=0)

        target_f[np.arange(batch_size), action_minibatch] = target

        self.model.fit(state_minibatch, target_f, epochs=batch_size, verbose=0)

        # for state, action, reward, next_state, done in minibatch:
        #     target = reward # If done, then target = reward
        #     if not done:
        #         target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

        #     target_f = self.model.predict(state, verbose=0)
        #     target_f[0][action] = target # Update the Q value for the prediction in the Q table

        #     self.model.fit(state, target_f, epochs=1, verbose=0)

        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def load(self, name: str) -> None:
        self.model.load_weights(name)

    def save(self, name: str) -> None:
        self.model.save_weights(name)