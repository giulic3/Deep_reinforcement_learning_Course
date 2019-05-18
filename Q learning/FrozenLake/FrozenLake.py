# import dependencies
import numpy as np
import gym
import random

# use the FrozenLake environment in OpenAI Gym
env = gym.make("FrozenLake-v0")

# retrieve number of rows (states) and columns (actions) to build the QTable
state_size = env.observation_space.n
action_size = env.action_space.n

# build the QTable initially filed with 0s using Numpy
qtable = np.zeros((state_size, action_size))
print(qtable)

# init hyperparameters
total_episodes = 15000        # total episodes
learning_rate = 0.8           # learning rate
max_steps = 99                # max steps per episode
gamma = 0.95                  # discounting rate

# exploration parameters
epsilon = 1.0                 # exploration rate
max_epsilon = 1.0             # exploration probability at start
min_epsilon = 0.01            # minimum exploration probability
decay_rate = 0.005            # exponential decay rate for exploration prob
