import threading
import torch
from datetime import datetime

import gym
import copy
import torch.nn.functional as functional
from torch import nn, optim
from torch.distributions import Categorical

CART_POLE = 'CartPole-v1'
EPISODES = 1000
SOLVED_SCORE = 200
GAMMA = 0.99
THREADS_NUMBER = 4
T_MAX = 5
ACTION_SPACE = 2
INPUT_SPACE = 4


class Network(nn.Module):
    def __init__(self, input_shape, action_space):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, action_space)
        self.optimizer = optim.Adam(self.parameters())
        self.value_loss = nn.MSELoss()

    def forward(self, input):
        x = functional.relu(self.fc1(input))
        x = functional.relu(self.fc2(x))
        value = self.output(x)
        probs = functional.sigmoid(value)
        return probs, value


def eval_state(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0).cuda()
    probs, value = policy(state)
    dist = Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob, value[0][action.item()]


class Actor(threading.Thread):
    def __init__(self, env, thread_num):
        threading.Thread.__init__(self)
        self.env = env
        self.train = True
        self.thread_num = thread_num

    def run(self):
        while self.train:
            thread_model = copy.deepcopy(network)
            thread_model.zero_grad()

            t = 0
            state = self.env.reset()

            done = False
            game_reward = 0
            episode = []

            while not done:
                action, logp, value = eval_state(thread_model, state)
                state, reward, done, _ = self.env.step(action)
                episode.append((state, action, reward, logp, value))
                game_reward += reward
                t += 1

                if done or t % T_MAX == 0:
                    self.update(thread_model, episode, done)

            append_episode_result(game_reward, self.thread_num)

    def update(self, thread_model, episode, done):
        r = 0 if done else episode[-1][4]
        policy_loss = []
        values = []
        rewards = []

        for state in episode[-T_MAX:]:
            r = GAMMA * r + state[2]
            policy_loss.append(-state[3] * r)
            rewards.append(r)
            values.append(state[4])

        # Gradient ascent for Policy
        thread_model.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        thread_model.optimizer.step()

        # Gradient descent for Value
        # thread_model.optimizer.zero_grad()
        # loss = thread_model.value_loss(torch.Tensor(values), torch.Tensor(rewards))
        # loss.backward()
        # thread_model.optimizer.step()

        update_network(thread_model)


def solved():
    return len(rewards) > 100 and sum(rewards[-100:]) / 100 >= SOLVED_SCORE


def train():
    for i in range(THREADS_NUMBER):
        env = gym.make(CART_POLE)
        thread = Actor(env, i)
        thread.start()
        threads.append(thread)


def append_episode_result(reward, thread_num):
    rewards.append(reward)
    print("Reward: " + str(reward) + " at " + str(datetime.now()) + " from thread: " + str(thread_num))
    if solved():
        for t in threads:
            t.train = False


def update_network(target):
    network_lock.acquire()
    network.load_state_dict(target.state_dict())
    network_lock.release()


rewards = []
threads = []
network = Network(INPUT_SPACE, ACTION_SPACE).cuda()
network_lock = threading.Lock()
train()

