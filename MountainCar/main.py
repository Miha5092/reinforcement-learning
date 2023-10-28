import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt

from agent import ContinuousAgent

env = gym.make('MountainCarContinuous-v0', render_mode='human')

state_size = env.observation_space.shape[0]

# Hyperparameters

initial_epsilon = 1.0
final_epsilon = 0.01
epsilon_decay = 0.997
gamma = 0.97
learning_rate = 0.004

agent = ContinuousAgent(state_size=state_size,
                 initial_epsilon=initial_epsilon,
                 final_epsilon=final_epsilon,
                 epsilon_decay=epsilon_decay,
                 gamma=gamma,
                 learning_rate=learning_rate)

def main():
    pass

if __name__ == '__main__':
    main()