import gym
import random
import pygame
import numpy as np
from ple import PLE
from ple.games import Snake
from collections import deque
from keras.layers import Dense, copy
from keras.models import Sequential


class QLearningHistory:
    def __init__(self, max_size=100):
        self.__max_size = max_size
        self.__history = deque(maxlen=max_size)
        self.size = 0

    def record_event(self, state, action, reward, new_state):
        if self.size < self.__max_size:
            self.size += 1
        self.__history.append((state, action, reward, new_state))

    def get_last_event(self):
        return self.__history[-1]

    def is_full(self):
        return self.size >= self.__max_size

    def get_events(self):
        return self.__history


class SnakeQNetwork:
    def __init__(self, food_reward=10,
                 dead_reward=-10,
                 alive_reward=2,
                 discount_factor=0.95,
                 batch_size=10,
                 train_epochs=100,
                 history_size=1000,
                 history_sample_size=50):
        self.food_reward = food_reward
        self.dead_reward = dead_reward
        self.alive_reward = alive_reward
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.history_size = history_size
        self.history_sample_size = history_sample_size
        self.q_learning_history = QLearningHistory(history_size)
        self.exploration_factor = 0.2
        self.next_move_prediction = None
        self.is_neural_network_initialized = False
        pygame.init()
        self.game = Snake(width=64, height=64)
        self.env = PLE(self.game, display_screen=True)
        self.env.init()
        self.LOG = gym.logger

    def run(self, maximum_number_of_iterations=10000, learning_rate=0.5, training=False):

        for iteration in range(0, maximum_number_of_iterations):

            if not self.is_neural_network_initialized:
                self.___initialize_neural_newtork()
                self.is_neural_network_initialized = True

            observation = self.env.getScreenGrayscale()
            observation_width = self.env.getScreenDims()[0]
            observation_height = self.env.getScreenDims()[1]
            self.game.init()

            # exit the while loop only if it's GAME OVER
            while True:

                q_values = self.next_move_prediction.predict(
                    x=observation.reshape(1, observation_width * observation_height),
                    batch_size=1)
                best_snake_action = np.argmax(q_values)
                reward = self.__take_snake_action(best_snake_action)
                previous_observation = copy.deepcopy(observation)
                observation = self.env.getScreenGrayscale()
                is_game_over = self.env.game_over()

                self.LOG.info("Current action reward: {r}. Is game over: {d}".format(r=reward, d=is_game_over))

                if training:
                    reward = self.__get_custom_reward(reward)

                    self.q_learning_history.record_event(
                        state=previous_observation, action=best_snake_action, reward=reward, new_state=observation)

                    last_event = self.q_learning_history.get_last_event()
                    self.LOG.info("Added event #{n} to history. Action: {a}; Reward: {r}"
                                  .format(a=last_event[1],
                                          r=reward,
                                          n=self.q_learning_history.size))

                    if self.q_learning_history.is_full():
                        history_batch = random.sample(self.q_learning_history.get_events(), self.history_sample_size)
                        self.LOG.info("Sampling {n} events from history.".format(n=self.history_sample_size))

                        training_batch_data = []
                        training_batch_labels = []

                        for history_event in history_batch:
                            old_state, action, reward, new_state = history_event

                            q_values_before_action = self.next_move_prediction.predict(
                                x=old_state.reshape(1, observation_width * observation_height), batch_size=1)

                            q_values_after_action = self.next_move_prediction.predict(
                                x=new_state.reshape(1, observation_width * observation_height), batch_size=1)

                            best_q_value_after_action = np.argmax(q_values_after_action)

                            training_q_values = np.zeros((1, 4))

                            for value_idx in range(0, len(q_values_before_action)):
                                training_q_values[value_idx] = q_values_before_action[value_idx]

                            output_update = learning_rate * (
                            reward + (self.discount_factor * best_q_value_after_action))

                            training_q_values[0][:] = 0
                            training_q_values[0][action] = output_update

                            training_batch_data.append(old_state.reshape(observation_width * observation_height, ))
                            training_batch_labels.append(training_q_values.reshape(4, ))

                        training_batch_data = np.array(training_batch_data)
                        training_batch_labels = np.array(training_batch_labels)

                        self.next_move_prediction.fit(
                            x=training_batch_data,
                            y=training_batch_labels,
                            epochs=self.train_epochs,
                            batch_size=self.batch_size)

                if is_game_over:
                    break

            if self.exploration_factor > 0.1:
                self.exploration_factor -= (1.0 / maximum_number_of_iterations)
                self.LOG.info("Exploration factor updated! New value: {v}".format(v=self.exploration_factor))

    def ___initialize_neural_newtork(self):
        input_layer_size = self.env.getScreenDims()[0] * self.env.getScreenDims()[1]
        hidden_layer_size = 100
        output_layer_size = 4

        input_layer = Dense(
            kernel_initializer='lecun_uniform',
            units=hidden_layer_size,
            input_shape=(input_layer_size,),
            activation='sigmoid')

        hidden_layer = Dense(
            kernel_initializer='lecun_uniform',
            units=output_layer_size,
            activation='linear'
        )

        self.next_move_prediction = Sequential()
        self.next_move_prediction.add(input_layer)
        self.next_move_prediction.add(hidden_layer)

        self.next_move_prediction.compile(
            optimizer='rmsprop',
            loss='mean_squared_error')

    def __take_snake_action(self, snake_action):

        random_number = np.random.random_sample()

        if not self.q_learning_history.is_full():
            snake_action = random.choice(self.env.getActionSet())
            self.LOG.info("Snake chose to do a random move - add to qHistory!")
            return self.env.act(snake_action)

        elif random_number < self.exploration_factor:
            snake_action = random.choice(self.env.getActionSet())
            self.LOG.info("Random number is smaller than exploration factor, {r} < {ef}! Snake chose random move!"
                          .format(r=random_number, ef=self.exploration_factor))
            return self.env.act(snake_action)

        elif snake_action == 0:
            self.LOG.info("Snake chose to go up")
            return self.env.act(115)

        elif snake_action == 1:
            self.LOG.info("Snake chose to go left")
            return self.env.act(97)

        elif snake_action == 2:
            self.LOG.info("Snake chose to go down")
            return self.env.act(119)

        elif snake_action == 3:
            self.LOG.info("Snake chose to go right")
            return self.env.act(100)

    def __get_custom_reward(self, reward):
        if reward >= 1:
            self.LOG.info("Has eaten food! Reward is {r}".format(r=self.food_reward))
            return self.food_reward
        elif reward >= 0:
            self.LOG.info("Stayed alive! Reward is {r}".format(r=self.alive_reward))
            return self.alive_reward
        else:
            self.LOG.info("Crashed! Reward is {r}".format(r=self.dead_reward))
            return self.dead_reward


if __name__ == "__main__":
    nn = SnakeQNetwork(
        food_reward=10,
        dead_reward=-10,
        alive_reward=2,
        discount_factor=0.3,
        batch_size=100,
        train_epochs=20,
        history_size=1000,
        history_sample_size=200
    )
    nn.run(
        maximum_number_of_iterations=100,
        learning_rate=0.8,
        training=True
    )
