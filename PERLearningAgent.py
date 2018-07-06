import numpy as np

from SumTree import SumTree
from DoubleQLearningAgent import DoubleQLearningAgent


class PERLearningAgent(DoubleQLearningAgent):

    def __init__(self, frames_shape, action_shape, memory_size):
        DoubleQLearningAgent.__init__(self, frames_shape, action_shape, 0)
        self.alpha = 0.6
        self.epsilon_positive = 0.01  # epsilon used to avoid not visiting states when p(i)=0
        self.memory = SumTree(memory_size)

    def remember(self, experience):
        self.memory.add(self.memory.max_p, experience)

    def replay(self, replay_length):
        replay_batch, idxs, weights = self.get_batch(replay_length)
        xs, ys, errors = self.get_targets(replay_batch)

        self.model.fit(xs, ys, batch_size=32, epochs=1, verbose=0, sample_weight=weights)
        self.update_epsilon()
        self.steps += 1

        if self.steps % self.target_update_freq == 0:
            self.update_target()

        # update priorities
        for i in range(replay_length):
            self.memory.update(idxs[i], errors[i])

    def get_targets(self, replay_batch):
        states = np.array([experience.state for experience in replay_batch])
        states_ = np.array([experience.next_state for experience in replay_batch])
        ys = np.zeros((len(replay_batch), self.action_space))
        errors = np.zeros(len(replay_batch))

        p = self.model.predict(states)
        p_ = self.model.predict(states_)
        p_target_ = self.target.predict(states_)

        for i in range(len(replay_batch)):
            experience = replay_batch[i]
            if not experience.done:
                target = experience.reward + self.discount * p_target_[i][np.argmax(p_[i])]
            else:
                target = experience.reward

            previous_prediction = p[i][experience.action]
            p[i][experience.action] = target
            ys[i] = p[i]
            errors[i] = abs(previous_prediction - target)

        return states, ys, errors

    def get_batch(self, batch_size):
        beta = 1 - self.discount
        max_priority = self.memory.total() / batch_size
        batch = []
        idxs = []
        weights = []

        for i in range(batch_size):
            low = max_priority * i
            high = max_priority * (i + 1)
            priority = np.random.uniform(low, high)
            (idx, p, exp) = self.memory.get(priority)
            weight = (len(self.memory) * p) ** (-beta)
            if exp is not None:
                batch.append(exp)
                idxs.append(idx)
                weights.append(weight / self.memory.max_p)

        return batch, idxs, np.array(weights)
