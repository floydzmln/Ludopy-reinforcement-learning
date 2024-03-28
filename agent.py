import tensorflow as tf
from tensorflow import keras
import numpy as np
import ludopy
from collections import deque
from ModifiedTensorboard import ModifiedTensorBoard
import time
import random

REPLAY_MEM_SIZE = 50_000
MIN_REPLAY_MEM_SIZE = 1_000
MINIBATCH_SIZE = 64  # how many samples to use for training
UPDATE_TARGET_EVERY = 5  # end of episodes
DISCOUNT = 0.99
LOAD_MODEL = "models/ConState____13.50max____6.35avg____0.90min__1711628047.model"

MODEL_NAME = "ConState"


class LudoAgent:
    def __init__(self):
        self.input_size = 9

        # main model, gets trained
        self.model = self.build_model()
        # target model, this is what we .predict against every step
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEM_SIZE)
        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/{MODEL_NAME}-{int(time.time())}"
        )

        self.target_update_counter = 0

    def build_model(self):
        """
        Input representation:
        0-240  location of players pieces
        241    number of safe zones occupied by players pieces
        241-421  opponents distances to finish
        421-427 dice
        Ouputs are four
        """
        if LOAD_MODEL is not None:
            print(f"Loading{LOAD_MODEL}")
            model = keras.models.load_model(LOAD_MODEL)
            print("loaded model")
        else:
            print("new model")
            output_size = 4
            # Define your NN architecture here (using Keras)
            model = keras.models.Sequential()
            model.add(keras.layers.Input(shape=(self.input_size, 1)))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(9, activation="linear"))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(6, activation="linear"))
            model.add(keras.layers.Dense(output_size, activation="linear"))
            model.compile(
                loss="mse",
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                metrics=["accuracy"],
            )
        return model

    def get_state(self, obs):
        (dice,move_pieces,player_pieces,enemy_pieces,player_is_a_winner,there_is_a_winner,) = obs
        # state representation: 4*player pieces, number of safe zones occupied, enemy pieces distance to finish, dice

        state_representation = np.zeros(self.input_size)
        state_representation[0]=player_pieces[0]/59
        state_representation[1]=player_pieces[1]/59
        state_representation[2]=player_pieces[2]/59
        state_representation[3]=player_pieces[3]/59
        safe_zones_occupied = 0
        safe_zones = [1, 9, 22, 35, 48, 53, 53, 54, 55, 56, 57, 58, 59]
        for i in range(4):
            for safe_tile in safe_zones:
                if safe_tile == player_pieces[i]:
                    safe_zones_occupied += 1
        state_representation[4] = safe_zones_occupied / 4
        state_representation[5] = max(enemy_pieces[0])/59
        state_representation[6] = max(enemy_pieces[1])/59
        state_representation[7] = max(enemy_pieces[2])/59
        state_representation[8] = dice/6

        """
        state_representation[player_pieces[0]] = 1
        state_representation[player_pieces[1] + 60] = 1
        state_representation[player_pieces[2] + 120] = 1
        state_representation[player_pieces[3] + 180] = 1
        # number of safe zones
        safe_zones_occupied = 0
        safe_zones = [1, 9, 22, 35, 48, 53, 53, 54, 55, 56, 57, 58, 59]
        for i in range(4):
            for safe_tile in safe_zones:
                if safe_tile == player_pieces[i]:
                    safe_zones_occupied += 1
        state_representation[240] = safe_zones_occupied / 4
        # furthest enemy pieces
        state_representation[max(enemy_pieces[0]) + 241] = 1
        state_representation[max(enemy_pieces[1]) + 301] = 1
        state_representation[max(enemy_pieces[2]) + 361] = 1

        state_representation[dice + 420] = 1
        """
        return state_representation

    def update_replay_memory(self, transition):
        self.replay_memory.append(
            transition
        )  # transition=(current_state, action, reward, new_state, there_is_a_winner)

    def get_qs(self, state):
        return self.model.predict(state.reshape(1, -1), verbose=0)

    def train(self, terminal_state, step):  # terminal_state=if there is a winner
        if len(self.replay_memory) < MIN_REPLAY_MEM_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current obs, then query for qs list
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states, verbose=0)

        new_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_states, verbose=0)

        X = []#features
        Y = []#labels
        # enumerate the batches
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            # if not a terminal state, get new q from future states, otherwise set it to 0
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            Y.append(current_qs)

        self.model.fit(
            np.array(X),
            np.array(Y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if terminal_state else None,
        )
        #updating to determine if we want to update target model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
