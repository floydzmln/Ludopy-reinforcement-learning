
import time
import pickle
import os
import numpy as np
from ModifiedTensorboard import ModifiedTensorBoard
import cv2
import ludopy
from tqdm import tqdm


EPISODES = 20000
EPSILON_DECAY = 0.99975  # adjust so that it fits with the total episodes your aiming for
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 100
MIN_REWARD = -1
MODEL_NAME = "QLudo_20k_lr0.001"
SHOW_BOARD = False
DISCOUNT = 0.99
LOAD_MODEL = None
TRAINING=True

ACTION_SPACE=4
STATE_SPACE=427

class LudoAgent:
    def __init__(self):
        if LOAD_MODEL is None:
            self.qtable={}
        else:
            f=open(LOAD_MODEL,"rb")
            self.qtable=pickle.load(f)
            f.close()
        self.learning_rate=0.001
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
    def get_state(self, obs):
        (
            dice,
            move_pieces,
            player_pieces,
            enemy_pieces,
            player_is_a_winner,
            there_is_a_winner,
        ) = obs
        # state representation: 4*player pieces, number of safe zones occupied, enemy pieces distance to finish, dice
        state={}
        locations=[0]*4
        locations[0]=player_pieces[0]/59
        locations[1]=player_pieces[1]/59
        locations[2]=player_pieces[2]/59
        locations[3]=player_pieces[3]/59
        state["locations"]=locations
        # number of safe zones
        safe_zones_occupied = 0
        safe_zones = [1, 9, 22, 35, 48, 53, 53, 54, 55, 56, 57, 58, 59]
        for i in range(4):
            for safe_tile in safe_zones:
                if safe_tile == player_pieces[i]:
                    safe_zones_occupied += 1

        state["safe_zones"] = safe_zones_occupied

        enemy_dist=[0]*3
        enemy_dist[0] = max(enemy_pieces[0])/59
        enemy_dist[1] = max(enemy_pieces[1])/59
        enemy_dist[2] = max(enemy_pieces[2])/59
        state["enemy_distances"]=enemy_dist
        state["dice_outcome"] = dice

        return self.abstract_ludo_state(state)
    
    def abstract_ludo_state(self,state):
        starting_zone = [0]
        safe_zone = [1, 9, 22, 35, 48, 53, 53]
        star_zone=[5,12,18,25,31,38,44,51] 
        home_track_zone = [54, 55, 56, 57, 58]
        finish=[59]
        # Abstracted state dictionary
        abstracted_state = {}
        # Piece location abstraction
        piece_locations = []
        for piece_loc in state["locations"]:
            if piece_loc in starting_zone:
                piece_locations.append("starting_zone")
            elif piece_loc in safe_zone:
                piece_locations.append("safe_zone")
            elif piece_loc in finish:
                piece_locations.append("finish")
            elif piece_loc in star_zone:
                piece_locations.append("star_zone")
            elif piece_loc in home_track_zone:
                for zone in home_track_zone:
                    if piece_loc in zone:
                        zone_index = home_track_zone[zone]
                        piece_locations.append(f"home_track_{zone_index}")
                    break
            else:
                piece_locations.append("default")

        abstracted_state["piece1_location"] = piece_locations[0]
        abstracted_state["piece2_location"] = piece_locations[1]
        abstracted_state["piece3_location"] = piece_locations[2]
        abstracted_state["piece4_location"] = piece_locations[3]
        # Enemy threat abstraction (replace with your threat categorization logic)
        enemy_threats = []
        for distance in state["enemy_distances"]:
            if distance <= 6:
                enemy_threats.append("close")
            elif distance <= 12:
                enemy_threats.append("mid")
            else:
                enemy_threats.append("far")
        abstracted_state["enemy1_threat"] = enemy_threats[0]
        abstracted_state["enemy2_threat"] = enemy_threats[1]
        abstracted_state["enemy3_threat"] = enemy_threats[2]
        # Add other features as needed (number of pieces in home track, dice outcome)
        abstracted_state["dice_outcome"] = state["dice_outcome"]
        return abstracted_state

    def get_qs(self,state):
        hashable_state=frozenset(state.items())
        if hashable_state not in self.qtable:
            self.qtable[hashable_state] = {action: 0 for action in self.get_possible_actions()}
        q=list(self.qtable[hashable_state].values())
        return q
    
    def train(self,state,action,reward,new_state):
        hashable_state=frozenset(state.items())
        hashable_new_state=frozenset(new_state.items())
        # Check if the state exists in the Q-table
        if hashable_state not in self.qtable:
            self.qtable[hashable_state] = {action: 0 for action in self.get_possible_actions()}
        
        if hashable_new_state not in self.qtable:
            self.qtable[hashable_new_state] = {action: 0 for action in self.get_possible_actions()}

        self.qtable[hashable_state][action] = self.qtable[hashable_state][action] + self.learning_rate * (
            reward + DISCOUNT * max(self.qtable[hashable_new_state].values()) - self.qtable[hashable_state][action])

    def get_possible_actions(self):
        possible_actions = [0,1,2,3]
        return possible_actions
    
    def calculate_td_error(self,state, action, reward, next_state):
        hashable_state=frozenset(state.items())
        hashable_new_state=frozenset(next_state.items())
        # Current Q-value
        current_q = self.qtable[hashable_state][action]   
        # Maximum Q-value for the next state
        max_next_q = np.max(list(self.qtable[hashable_new_state].values()))   
        # TD error
        td_error = reward + DISCOUNT * max_next_q - current_q
        return td_error


def get_reward(new_obs, current_obs):
    reward = 0
    (_, _, old_player_pieces, old_enemy_pieces, _, _) = current_obs[0]
    (_, _, new_player_pieces, new_enemy_pieces, player_is_a_winner, _) = new_obs
    if player_is_a_winner:
        reward += 4

    for i in range(4):
        if old_player_pieces[i] == 0 and new_player_pieces[i] != 0:  # move out of home
            reward += 0.25
        if old_player_pieces[i] != 0 and new_player_pieces[i] == 0:  # piece died
            reward -= 1
        if (
            old_player_pieces[i] <= 53 and new_player_pieces[i] >= 54
        ):  # move into goal area
            reward += 0.4
        if old_player_pieces[i] != 59 and new_player_pieces[i] == 59:  # move to goal
            reward += 1

        for j in range(3):
            # killed other piece
            if (
                old_player_pieces[i] != old_enemy_pieces[j][i]
                and old_enemy_pieces[j][i] != 0
                and new_player_pieces[i] == old_enemy_pieces[j][i]
                and new_enemy_pieces[j][i] == 0
            ):
                reward += 0.4

    return reward


def move_rand(move_pieces):
    if len(move_pieces):  # Can be zero when player doesnt get a six to move piece out
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = np.random.randint(0,3)
    return piece_to_move


def get_best_valid_action(qs, move_pieces):  # Is there a better way to do this?
    if len(move_pieces) > 1:
        data = [(0, qs[0]), (1, qs[1]), (2, qs[2]), (3, qs[3])]
        try:
            data.sort(key=lambda x: x[1])
        except ValueError:
            pass
        data.reverse()
        for entry in data:
            if entry[0] in move_pieces:
                return entry[0]
    elif len(move_pieces) == 1:
        return move_pieces[0]
    else:
        return np.random.randint(0,3)


def single_game(g, step, epsilon):
    g.reset()
    there_is_a_winner = False
    episode_reward = 0
    action = 0

    while not there_is_a_winner:
        current_obs = g.get_observation()
        (
            dice,
            move_pieces,
            player_pieces,
            enemy_pieces,
            player_is_a_winner,
            there_is_a_winner,
        ), player_i = current_obs

        if player_i != 0:  # other players turn
            _, _, _, _, _, there_is_a_winner = g.answer_observation(move_rand(move_pieces))
            continue
        else:  # agent player
            if np.random.random() > epsilon:
                current_state = agent.get_state(current_obs[0])  # translate game state
                qs = agent.get_qs(current_state)  
                action = get_best_valid_action(qs, move_pieces)

            else:
                action = move_rand(move_pieces)  # random move

            new_obs = g.answer_observation(action)  # new state
            reward = get_reward(new_obs, current_obs)
            # Transform new continous state to new discrete state and count reward
            episode_reward += reward
            # optionally show board
            if SHOW_BOARD:
                cv2.imshow("board", g.render_environment())
                cv2.waitKey(0)

            if TRAINING:
                agent.train(agent.get_state(current_obs[0]),action,reward,agent.get_state(new_obs))
                td_error=agent.calculate_td_error(agent.get_state(current_obs[0]),action,reward,agent.get_state(new_obs))
            current_obs = new_obs
            step += 1

    # print(player_pieces, player_i, enemy_pieces)
    # print("Saving history to numpy file")
    # g.save_hist("game_history.npy")
    # print("Saving game video")
    # g.save_hist_video("game_video.mp4")

    return episode_reward, step, g.get_winner_of_game(), td_error


if __name__ == "__main__":
    # Create models folder
    if not os.path.isdir("tables"):
        os.makedirs("tables")
    agent = LudoAgent()
    ghosts = []
    wins = 0
    ep_rewards = [0]
    epsilon = 0
    if TRAINING:
        epsilon = 1
    td_errors=[]
    start_time = time.time()
    g = ludopy.Game(ghost_players=ghosts)
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1

        episode_reward, step, winner_of_game,td_error = single_game(g, step, epsilon)

        td_errors.append(td_error**2)

        if winner_of_game == 0:
            wins += 1
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)

        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            mean_squared_td_error = np.mean(td_errors)
            td_errors.clear()
            win_rate = wins / AGGREGATE_STATS_EVERY
            wins = 0
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(
                reward_avg=average_reward,
                reward_min=min_reward,
                reward_max=max_reward,
                epsilon=epsilon,
                win_rate=win_rate,
                mean_squared_td_error=mean_squared_td_error
            )

            # Save model, but only when min reward (or average reward) is greater or equal to a set value
            if episode==EPISODES:
                f=open(f"tables/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.pkl","wb")
                pickle.dump(agent.qtable,f)
                f.close()

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
