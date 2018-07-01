import datetime

import gym
import numpy
import scipy.misc

from DoubleQLearningAgent import DoubleQLearningAgent
from QLearningAgent import QLearningAgent, Experience

EPISODES = 5000
TARGET_UPDATE_FREQ = 7500
OBSERVE_LIMIT = 100000
REPLAY_SIZE = 32
STATE_STACK_SIZE = 4
IMAGE_SIZE = 84

PACMAN = 'MsPacman-v0'
CARTPOLE = 'CartPole-v1'
PONG = 'Pong-v0'


def process(state):
    grayscale_state = numpy.dot(state[..., :3], [0.299, 0.587, .114])
    reshaped_image = scipy.misc.imresize(grayscale_state, (IMAGE_SIZE, IMAGE_SIZE))
    return numpy.reshape(reshaped_image, (IMAGE_SIZE, IMAGE_SIZE))


def train_agent(env, agent):
    agent.load_agent()

    for ep in range(EPISODES):
        state = process(env.reset())

        done = False
        game_reward = 0
        frames = 0
        state_stack = numpy.array([state, state, state, state])

        while not done:
            action = agent.act(state_stack)
            next_state, reward, done, _ = env.step(action)
            reward = numpy.clip(reward, -1, 1)

            state = numpy.array(state_stack)
            state_stack = numpy.roll(state_stack, STATE_STACK_SIZE - 1)
            state_stack[STATE_STACK_SIZE - 1] = process(next_state)
            state_ = numpy.array(state_stack)

            agent.remember(Experience(state, action, reward, state_, done))

            if len(agent.memory) >= OBSERVE_LIMIT:
                agent.replay(REPLAY_SIZE)
            if frames % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

            frames += 1
            game_reward += reward

        print("Reward: " + str(game_reward) + " with frames: " + str(frames) + " at " + str(datetime.datetime.now())
              + " with epsilon: " + str(agent.epsilon))


def use_agent(env, agent):
    agent.load_agent()
    state = process(env.reset())
    done = False
    state_stack = numpy.array([state, state, state, state])

    for i in range(EPISODES):
        while not done:
            action = agent.act_best_action(state_stack)
            next_state, _, done, _ = env.step(action)

            state_stack = numpy.roll(state_stack, STATE_STACK_SIZE - 1)
            state_stack[STATE_STACK_SIZE - 1] = process(next_state)

            env.render()
        env.reset()
        done = False


env = gym.make(PONG)
input_shape = (STATE_STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE)
agent = DoubleQLearningAgent(input_shape, env.action_space.n, OBSERVE_LIMIT)
# PERLearningAgent(input_shape, env.action_space.n, OBSERVE_LIMIT)
# QLearningAgent(input_shape, env.action_space.n, OBSERVE_LIMIT)

try:
    train_agent(env, agent)
finally:
    agent.save_agent()

# use_agent(env, agent)
