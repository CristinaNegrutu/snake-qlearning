# Snake
import keras
import random
import gym
from keras.layers import Dense, copy
from keras.models import Sequential
from ple.games import Snake
from ple import PLE
import pygame
import numpy as np

import random
from collections import deque

import gym_ple  # Do not delete, even if IDE says it's not used

from utils import GrayscaleConverter, ImageResizer


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
    def __init__(self, food_reward=1, dead_reward=-1, alive_reward=0, discount_factor=0.95, nn_batch_size=10,
                 nn_train_epochs=5, nn_history_size=100, nn_history_sample_size=50):

        self.LOG = gym.logger
        self.nn_batch_size = nn_batch_size
        self.nn_train_epochs = nn_train_epochs

        self.nn_history_size = nn_history_size
        self.nn_history_sample_size = nn_history_sample_size
        self.food_reward = food_reward
        self.dead_reward = dead_reward
        self.alive_reward = alive_reward
        self.discount_factor = discount_factor
        self.q_learning_history = QLearningHistory(self.nn_history_size)
        self.exploration_factor = 1

        self.state_prediction_nn = None
        self.nn_initialized = False
        pygame.init()

        game = Snake(width=200, height=200)
        game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
        game.clock = pygame.time.Clock()
        game.rng = np.random.RandomState(24)
        self.game = game
        env = PLE(self.game, display_screen=True, fps=30)
        env.init()
        self.env = env

    def run(self, episode_count=1000, learning_rate=0.5, training=False):
        for episode_idx in range(0, episode_count):
            self.LOG.info("Episode #{n} started.".format(n=episode_idx))
            self.game.init()
            observation = self.env.getScreenGrayscale()
            # observation_object = self.__downscale_image(observation)

            if not self.nn_initialized:
                self.__initialize_nn()
                self.nn_initialized = True
                self.LOG.info("Initialized the neural network!")

            observation_width = self.env.getScreenDims()[0]
            observation_height = self.env.getScreenDims()[1]

            while True:
                q_values = self.state_prediction_nn.predict(
                    x=observation.reshape(1, observation_width * observation_height),
                    batch_size=1)

                old_observation = copy.deepcopy(observation)
                print "q_values ", q_values
                snake_action = (np.argmax(q_values))  # the predicted action
                index_of_action_in_q_values = np.where(q_values == snake_action)
                observation = self.env.getScreenGrayscale()
                # observation = self.__downscale_image(observation)
                reward = self.__take_snake_action(index_of_action_in_q_values)

                dt = self.game.clock.tick_busy_loop(100)
                self.game.step(dt)
                pygame.display.update()
                print "\n", self.game.getGameState() ,"\n"

                done = self.env.game_over()
                self.LOG.info("Current action reward: {r}. Done: {d}".format(r=reward, d=done))

                if training:
                    reward = self.__get_custom_reward(reward)

                    self.q_learning_history.record_event(
                        state=old_observation, action=snake_action, reward=reward, new_state=observation)

                    last_event = self.q_learning_history.get_last_event()
                    self.LOG.info("Added event #{n} to history. Action: {a}; Reward: {r}"
                                  .format(a=last_event[1],
                                          r=reward,
                                          n=self.q_learning_history.size))

                    if self.q_learning_history.is_full():
                        history_batch = random.sample(self.q_learning_history.get_events(), self.nn_history_sample_size)
                        self.LOG.info("Sampling {n} events from history with size {s}"
                                      .format(n=self.nn_history_sample_size, s=self.q_learning_history.size))

                        nn_training_batch_data = []
                        nn_training_batch_labels = []

                        for history_event in history_batch:
                            old_state, action, reward, new_state = history_event

                            q_values_before_action = self.state_prediction_nn.predict(
                                x=old_state.reshape(1, observation_width * observation_height), batch_size=1)

                            q_values_after_action = self.state_prediction_nn.predict(
                                x=new_state.reshape(1, observation_width * observation_height), batch_size=1)

                            best_q_value_after_action = np.max(q_values_after_action)

                            training_q_values = np.zeros((1, 4))  # 4 possible actions

                            for value_idx in range(0, len(q_values_before_action)):
                                training_q_values[value_idx] = q_values_before_action[value_idx]

                            output_update = learning_rate * (reward + (self.discount_factor * best_q_value_after_action))

                            training_q_values[0][0] = 0
                            training_q_values[0][1] = 0
                            training_q_values[0][2] = 0
                            training_q_values[0][3] = 0

                            # print "action.......", action
                            training_q_values[0][action] = output_update

                            nn_training_batch_data.append(old_state.reshape(observation_width * observation_height, ))
                            nn_training_batch_labels.append(training_q_values.reshape(4, ))

                        nn_training_batch_data = np.array(nn_training_batch_data)
                        nn_training_batch_labels = np.array(nn_training_batch_labels)

                        self.state_prediction_nn.fit(
                            x=nn_training_batch_data,
                            y=nn_training_batch_labels,
                            epochs=self.nn_train_epochs,
                            batch_size=self.nn_batch_size)
                if done:
                    break
            if self.exploration_factor > 0.1:
                self.exploration_factor -= (1.0 / episode_count)
                self.LOG.info("Exploration factor updated! New value: {v}".format(v=self.exploration_factor))

        self.env.act(None)

    def __downscale_image(self, image):
        """
        Resized the input image and converts it to grayscale
        :param image: The image do be downsized and grayscaled
        :return: The grayscale, resized image corresponding to the input image
        """
        grayscale_observation_image = GrayscaleConverter.rgb_to_grayscale(image)
        resized_observation_image = ImageResizer.resize_image(
            image=grayscale_observation_image,
            ratio=0.5)
        return resized_observation_image

    def __initialize_nn(self):
        nn_input_layer_size = self.env.getScreenDims()[0] * self.env.getScreenDims()[1]
        nn_hidden_layer_size = 100
        nn_output_layer_size = 4  # 1 possible action outcome

        nn_input_layer = Dense(
            kernel_initializer='lecun_uniform',  # Uniform initialization scaled by the square root of the number of inputs
            units=nn_hidden_layer_size,
            input_shape=(nn_input_layer_size,),
            activation='sigmoid')

        self.LOG.info("Adding layer to neural network: input_size: {i}, output_size: {o}"
                      .format(i=nn_input_layer_size, o=nn_hidden_layer_size))

        nn_hidden_layer = Dense(
            kernel_initializer='lecun_uniform',
            units=nn_output_layer_size,
            activation='linear'  # Pass value along -> f(x) = x
        )

        self.LOG.info("Adding layer to neural network: output_size: {o}"
                      .format(o=nn_output_layer_size))

        self.state_prediction_nn = Sequential()  # Initialize a nn with a linear stack of layers
        self.state_prediction_nn.add(nn_input_layer)
        self.state_prediction_nn.add(nn_hidden_layer)

        # use mean squared error regression (aka cost derivative)
        # to compute errors to propagate backwards
        self.state_prediction_nn.compile(
            optimizer='rmsprop',
            loss='mean_squared_error')

    def __take_snake_action(self, snake_action):
        random_number = np.random.random_sample()
        if not self.q_learning_history.is_full():
            # print "index:::", self.env.getActionSet().index(snake_action)
            snake_action = self.env.getActionSet().index(random.choice(self.env.getActionSet()))
            self.LOG.info("Snake chose to do a random move - add to qHistory!")
            print "action set:::", self.env.getActionSet()
            # print "snake action:: ", snake_action
            return self.env.act(snake_action)

        elif random_number < self.exploration_factor:
            snake_action = self.env.getActionSet().index(random.choice(self.env.getActionSet()))
            self.LOG.info("Epsilon strikes rand={r} < {ef}! Snake chose random move!"
                          .format(r=random_number, ef=self.exploration_factor))
            return self.env.act(snake_action)

        elif snake_action == 0:
            self.LOG.info("Snake chose to go up")
            return self.env.act(snake_action)

        elif snake_action == 3:
            self.LOG.info("Snake chose to go down")
            return self.env.act(snake_action)

        elif snake_action == 2:
            self.LOG.info("Snake chose to go right")
            return self.env.act(snake_action)

        elif snake_action == 1:
            self.LOG.info("Snake chose to go left")
            return self.env.act(snake_action)

    def __get_custom_reward(self, reward):
        if reward >= 1:
            self.LOG.info("Has eaten food! -> Reward is {r}".format(r=reward))
            return self.food_reward
        elif reward >= 0:
            self.LOG.info("Stayed alive! -> Reward is {r}".format(r=reward))
            return self.alive_reward
        else:
            self.LOG.info("Crashed! -> Reward is {r}".format(r=reward))
            return self.dead_reward


if __name__ == "__main__":
    nn = SnakeQNetwork(
        food_reward=50,
        dead_reward=-1000,
        alive_reward=1,
        discount_factor=0.3,  # a future reward is more important than a proximity reward, so closer to 1.
        nn_batch_size=50,
        nn_train_epochs=5,
        # nn_image_resize_ratio=0.25,
        nn_history_size=100,
        nn_history_sample_size=50
    )
    nn.run(
        episode_count=100,
        learning_rate=0.8,
        training=True
    )
