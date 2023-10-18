import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt

from agent import DQNAgent

env = gym.make('CartPole-v1', render_mode='human')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters

n_episodes = 400
batch_size = 64

initial_epsilon = 1.0
final_epsilon = 0.01
epsilon_decay = 0.994
gamma = 0.95
learning_rate = 0.01

agent = DQNAgent(state_size=state_size,
                 action_size=action_size,
                 initial_epsilon=initial_epsilon,
                 final_epsilon=final_epsilon,
                 epsilon_decay=epsilon_decay,
                 gamma=gamma,
                 learning_rate=learning_rate)

def main():
    scores = []

    for episode in range(n_episodes):
        state = env.reset()
        state = np.reshape(state[0], [1, state_size])

        done = False
        score = 0
        while not done:
            # env.render()
            
            action = agent.act(state)

            next_state, reward, done, _, _ = env.step(action)

            reward = reward if not done else -10

            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            score += reward
            if done:
                print(f'Episode: {episode + 1}/{n_episodes}, score: {score}, epsilon: {agent.epsilon:.2f}')
                scores.append(score)

        # Once enough data is gathered, train the agent
        if len(agent.state_memory) > batch_size:
            agent.replay(batch_size)

    env.close()

    # Save the model
    agent.model.save('model.h5')

    window_size = 15  # Adjust this to your desired window size

    # Calculate the moving average
    moving_average = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')

    # Plot the moving average
    plt.plot(moving_average)
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Score')
    plt.show()


if __name__ == '__main__':
    main()