import copy
import sys
import time
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from agent import LudoAgent
import cv2
import ludopy
from tqdm import tqdm



EPISODES=200
EPSILON_DECAY=0.99975
MIN_EPSILON=0.001
AGGREGATE_STATS_EVERY=50
MIN_REWARD=-1
MODEL_NAME="427x1"
SHOW_BOARD=False

def get_reward(new_obs,current_obs):
    """This function returns the reward of the game, based on the observations.

    :param old_observations: The observations of the past state, before making a decition
    :param new_observations: The observations of the current state, after making a decition

    :return: The reward of the game"""
    reward = 0
    (_, _, old_player_pieces, old_enemy_pieces, _, _) = current_obs[0]
    (_, _, new_player_pieces, new_enemy_pieces, player_is_a_winner, _) = new_obs
    if player_is_a_winner:
        reward += 4
    
    for i in range(4):
        if old_player_pieces[i] == 0 and new_player_pieces[i] != 0:#move out of home
            reward += 0.25
        if old_player_pieces[i] != 0 and new_player_pieces[i] == 0:# piece died
            reward -= 1
        if old_player_pieces[i] <= 53 and new_player_pieces[i] >= 54:#move into goal area
            reward += 0.4
        if old_player_pieces[i] != 59 and new_player_pieces[i] == 59:#move to goal
            reward += 1
        
        for j in range(3):
            #killed other piece
            if old_player_pieces[i] != old_enemy_pieces[j][i] and old_enemy_pieces[j][i] != 0 and new_player_pieces[i] == old_enemy_pieces[j][i] and new_enemy_pieces[j][i] == 0:
                reward += 0.4
            
    return reward

def move_rand(move_pieces):
    if len(move_pieces):#Can be zero when player doesnt get a six to move piece out
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = -1
    return piece_to_move

def get_best_valid_action(qs, move_pieces):
  if len(move_pieces)>1:
    data=[(qs[0,1],0),(qs[0,1],1),(qs[0,2],2),(qs[0,3],3)]
    data.sort(key=lambda x: x[0])
    data.reverse()
    for entry in data:
      if entry[1] in move_pieces:
        return entry[1]
  elif len(move_pieces)==1:
    return move_pieces[0]
  else:
    return -1


def single_game(g,step,epsilon):
    g.reset()
    there_is_a_winner = False
    episode_reward=0
    action = 0
    overshoots = 0

    while not there_is_a_winner:
        current_obs=g.get_observation()
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = current_obs

        
        if (player_i != 0):#other players turn  
            _, _, _, _, _, there_is_a_winner = g.answer_observation(move_rand(move_pieces))
            continue
        else:# agent player
            if np.random.random() > epsilon:
                current_state=agent.get_state(current_obs[0])#translate game state
                qs=(agent.get_qs(current_state))#predict
                action=get_best_valid_action(qs,move_pieces)

            else:
                action=move_rand(move_pieces)#random move

            new_obs=g.answer_observation(action)#new state
            reward=get_reward(new_obs,current_obs)
            # Transform new continous state to new discrete state and count reward
            episode_reward += reward
            #optionally show board
            if SHOW_BOARD:
                cv2.imshow("board",g.render_environment())
                cv2.waitKey(0)

            # Every step we update replay memory and train main network
            agent.update_replay_memory((agent.get_state(current_obs[0]), action, reward,
                                         agent.get_state(new_obs), there_is_a_winner))
            agent.train(there_is_a_winner, step)
            current_obs = new_obs
            step += 1

    #print(player_pieces, player_i, enemy_pieces)
    #print("Saving history to numpy file")
    #g.save_hist("game_history.npy")
    #print("Saving game video")
    #g.save_hist_video("game_video.mp4")

    return episode_reward, step, g.get_winner_of_game(), overshoots

if __name__=="__main__":
    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')
    agent=LudoAgent()
    ghosts = []
    wins=0
    ep_rewards=[-200]
    epsilon=1#TODO change back to 1
    start_time = time.time()
    g = ludopy.Game(ghost_players=ghosts)
    for episode in tqdm(range(1,EPISODES+1),ascii=True, unit="episode"):
        agent.tensorboard.step=episode
        episode_reward=0
        step=1

        episode_reward, step, winner_of_game, overshoots=single_game(g,step,epsilon)
        if winner_of_game==0:
            wins+=1
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)

        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            win_rate=wins/AGGREGATE_STATS_EVERY
            wins=0
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, win_rate=win_rate)

            # Save model, but only when min reward (or average reward) is greater or equal to a set value
            if average_reward >= 4:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)



