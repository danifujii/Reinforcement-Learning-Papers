import datetime

import gym
import numpy

from DoubleQLearningAgent import DoubleQLearningAgent
from PERLearningAgent import PERLearningAgent
from QLearningAgent import Experience, QLearningAgent

EPISODES = 3000
TARGET_UPDATE_FREQ = 7500
OBSERVE_LIMIT = 50000
MEMORY_SIZE = 50000
REPLAY_SIZE = 32
STATE_STACK_SIZE = 4
IMAGE_SIZE = 84


CARTPOLE = 'CartPole-v1'
MOUNTAIN_CLIMBER = 'MountainCar-v0'


def solved(rewards):
    return len(rewards) > 100 and sum(rewards[-100:]) / 100 >= 200


def train_agent(env, agent):
    # agent.load_agent()
    frames = 0
    rewards = []

    for ep in range(EPISODES):
        state = env.reset()

        done = False
        game_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = numpy.clip(reward, -1, 1)

            agent.remember(Experience(state, action, reward, next_state, done))
            state = next_state

            if len(agent.memory) >= OBSERVE_LIMIT:
                agent.replay(REPLAY_SIZE)

            frames += 1
            game_reward += reward

        print("Reward: " + str(game_reward) + " with frames: " + str(frames) + " at " + str(datetime.datetime.now())
              + " with epsilon: " + str(agent.epsilon))

        rewards.append(game_reward)
        if solved(rewards):
            break


def use_agent(env, agent):
    agent.load_agent()
    state = env.reset()
    done = False

    for i in range(EPISODES):
        while not done:
            action = agent.act_best_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
            env.render()
        env.reset()
        done = False


env = gym.make(CARTPOLE)
input_shape = env.observation_space.shape
agent = PERLearningAgent(input_shape, env.action_space.n, MEMORY_SIZE)

# Possible learning algorithms:
# DoubleQLearningAgent(input_shape, env.action_space.n, OBSERVE_LIMIT)
# PERLearningAgent(input_shape, env.action_space.n, OBSERVE_LIMIT)
# QLearningAgent(input_shape, env.action_space.n, OBSERVE_LIMIT)

train = True

if train:
    try:
        train_agent(env, agent)
    finally:
        agent.save_agent()
else:
    use_agent(env, agent)
