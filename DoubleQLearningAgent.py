import math

import numpy as np

from QLearningAgent import QLearningAgent


class DoubleQLearningAgent(QLearningAgent):
    steps = 0
    epsilon_max = 1.0
    EXPLORATION_STOP = 500000
    LAMBDA = - math.log(0.01) / EXPLORATION_STOP

    def remember(self, experience):
        super().remember(experience)
        self.steps += 1

    def replay(self, replay_length):
        replay_batch = self.get_batch(replay_length)
        xs, ys = self.get_targets(replay_batch)
        self.model.fit(xs, ys, batch_size=32, epochs=1, verbose=0)
        self.update_epsilon()

    def get_targets(self, replay_batch):
        states = np.array([experience.state for experience in replay_batch])
        states_ = np.array([experience.next_state for experience in replay_batch])
        ys = np.zeros((len(replay_batch), self.action_space))

        p = self.model.predict(states)
        p_ = self.model.predict(states_)
        p_target_ = self.target.predict(states_)

        for i in range(len(replay_batch)):
            experience = replay_batch[i]

            if not experience.done:
                target = experience.reward + self.discount * p_target_[i][np.argmax(p_[i])]
            else:
                target = experience.reward

            p[i][experience.action] = target
            ys[i] = p[i]

        return states, ys

    def get_batch(self, batch_size):
        return np.random.choice(self.memory, size=batch_size)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_min + \
                           (self.epsilon_max - self.epsilon_min) * math.exp(-self.LAMBDA * self.steps)
