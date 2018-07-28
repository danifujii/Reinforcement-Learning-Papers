import threading
from datetime import datetime

import gym
import torch
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
        self.value = nn.Linear(32, 1)
        self.probs = nn.Linear(32, action_space)

        self.value_loss = nn.MSELoss()
        self.input_shape = input_shape
        self.output_shape = action_space

    def forward(self, input):
        x = functional.relu(self.fc1(input))
        x = functional.relu(self.fc2(x))
        value = self.value(x)
        probs = functional.sigmoid(self.probs(x))
        return probs, value


def eval_state(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0).cuda()
    probs, value = policy(state)
    dist = Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob, value


class Actor(threading.Thread):
    def __init__(self, env, thread_num, common_network):
        threading.Thread.__init__(self)
        self.env = env
        self.train = True
        self.thread_num = thread_num
        self.thread_network = Network(common_network.input_shape, common_network.output_shape).cuda()
        self.global_network = common_network

        self.env.reset()

    def run(self):
        t = game_reward = 0
        done = False
        episode = []
        state = self.env.reset()

        while self.train:
            self.thread_network.load_state_dict(self.global_network.state_dict())  # Sync weights to global parameters
            t_start = t

            while not done and t - t_start != T_MAX:
                action, logp, value = eval_state(self.thread_network, state)
                state, reward, done, _ = self.env.step(action)
                episode.append((state, action, reward, logp, value))
                game_reward += reward
                t += 1

            self.update(episode, done, t_start)

            if done:
                # Logging
                append_episode_result(game_reward, self.thread_num)

                # Clean up
                t = game_reward = 0
                done = False
                episode.clear()
                state = self.env.reset()

    def update(self, episode, done, t_start):
        r = 0 if done else episode[-1][4]
        policy_loss = []
        value_loss = []

        for t in reversed(range(t_start, len(episode) - 1)):
            state = episode[t]
            r = GAMMA * r + state[2]
            value = state[4]
            policy_loss.append(-state[3] * (r - value))
            value_loss.append((r - value) ** 2)

        if len(policy_loss) == 0:
            return

        loss = torch.cat(policy_loss).sum() + torch.stack(value_loss).mean()
        update_network(self.global_network, self.thread_network, loss)


def solved():
    return len(rewards) > 100 and sum(rewards[-100:]) / 100 >= SOLVED_SCORE


def train():
    for i in range(THREADS_NUMBER):
        env = gym.make(CART_POLE)
        thread = Actor(env, i, network)
        thread.start()
        threads.append(thread)


def append_episode_result(reward, thread_num):
    rewards.append(reward)
    print("Reward: " + str(reward) + " at " + str(datetime.now()) + " from thread: " + str(thread_num))
    if solved():
        for t in threads:
            t.train = False


def update_network(global_network, thread_network, loss):
    network_lock.acquire()
    shared_optimizer.zero_grad()
    loss.backward()
    for thread_param, global_param in zip(thread_network.parameters(), global_network.parameters()):
        global_param.grad = thread_param.grad
    shared_optimizer.step()
    network_lock.release()


rewards = []
threads = []
network = Network(INPUT_SPACE, ACTION_SPACE).cuda()
shared_optimizer = optim.Adam(network.parameters())
network_lock = threading.Lock()
train()

