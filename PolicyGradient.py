from datetime import datetime

import gym
import torch
import torch.nn.functional as functional
import torch.nn as nn

CART_POLE = 'CartPole-v1'
EPISODES = 1000
SOLVED_SCORE = 200


class Policy(nn.Module):
    def __init__(self, input_shape, action_space):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, action_space)

    def forward(self, input):
        x = functional.relu(self.fc1(input))
        x = functional.relu(self.fc2(x))
        return functional.sigmoid(self.output(x))


def solved(rewards):
    return len(rewards) > 100 and sum(rewards[-100:]) / 100 >= SOLVED_SCORE


def act(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0).cuda()
    probs = policy(state)
    return torch.argmax(probs).item()


def train(env, policy):
    rewards = []

    for ep in range(EPISODES):
        state = env.reset()

        done = False
        game_reward = 0
        episode = []

        while not done:
            action = act(policy, state)
            _, reward, done, _ = env.step(action)
            episode.append((state, action, reward))

            game_reward += reward

        print("Reward: " + str(game_reward) + " at " + str(datetime.now()))

        rewards.append(game_reward)
        if solved(rewards):
            break


env = gym.make(CART_POLE)
policy = Policy(env.observation_space.shape[0], env.action_space.n).cuda()
train(env, policy)
