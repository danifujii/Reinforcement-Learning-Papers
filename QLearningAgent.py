from collections import deque

import numpy as np
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential


class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class QLearningAgent:
    model_name = 'dqn_weights3.h5'
    discount = 0.99
    epsilon_min = 0.01

    def __init__(self, frames_shape, action_space, memory_size):
        self.frames_shape = frames_shape
        self.action_space = action_space
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=memory_size)
        self.model = self._build_image_model()
        self.target = self._build_image_model()
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
        model.compile(loss='mse', optimizer='rmsprop')
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
        replay_batch = np.random.choice(self.memory, size=replay_length)
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

        # epsilon decay
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
