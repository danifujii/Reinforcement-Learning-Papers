import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras import backend as K

HUBER_LOSS_DELTA = 1.0


def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)

    return K.mean(loss)


class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class QLearningAgent:
    model_name = 'dqn_weights3.h5'
    target_update_freq = 10000
    discount = 0.99
    epsilon_min = 0.01
    steps = 0

    def __init__(self, frames_shape, action_space, memory_size):
        self.frames_shape = frames_shape
        self.action_space = action_space
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=memory_size)
        self.model = self._build_model()    # self._build_image_model()
        self.target = self._build_model()   # self._build_image_model()
        self.update_target()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=self.frames_shape))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def _build_image_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=self.frames_shape,
                         data_format="channels_first"))
        model.add(Conv2D(32, (4, 4), strides=2, activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def update_target(self):
        self.target.set_weights(self.model.get_weights())

    def remember(self, experience):
        self.memory.append(experience)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(low=0, high=self.action_space)
        else:
            state = np.reshape(state, (1,) + state.shape)
            predictions = self.model.predict(state)[0]
            best_action = np.argmax(predictions)
            return best_action

    def replay(self, replay_length):
        replay_batch = self.get_batch(replay_length)
        xs, ys = [], []

        states = np.array([experience.state for experience in replay_batch])
        states_ = np.array([experience.next_state for experience in replay_batch])

        p = self.model.predict(states)
        p_ = self.target.predict(states_)

        for i in range(replay_length):
            experience = replay_batch[i]
            if not experience.done:
                target = experience.reward + self.discount * np.amax(p_[i])
            else:
                target = experience.reward

            prediction = p[i]
            prediction[experience.action] = target

            xs.append(experience.state)
            ys.append(prediction)

        # model training
        self.model.fit(np.array(xs), np.array(ys), batch_size=32, epochs=1, verbose=0)

        self.update_epsilon()
        self.steps += 1

        if self.steps % self.target_update_freq == 0:
            self.update_target()

    def get_batch(self, batch_size):
        idxs = [random.randint(0, len(self.memory) - 1) for _ in range(batch_size)]
        batch = [self.memory[i] for i in idxs]
        return batch

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def load_agent(self):
        self.model.load_weights(self.model_name)
        self.update_target()

    def save_agent(self):
        self.target.save_weights(self.model_name)

    def act_best_action(self, state):
        state = np.reshape(state, (1,) + state.shape)
        return np.argmax(self.model.predict(state)[0])
