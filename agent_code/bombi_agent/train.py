import os
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e


# This is only an example!
from agent_code.bombi_agent.feature_extraction import RLFeatureExtraction

## TRAINING PARAMETERS

# Allow to check various combinations of training methods in parallel
# through multiple agent processes by taking values from the
# environment.
t_policy = os.environ.get('BOMBI_POLICY')
t_policy_eps = os.environ.get('BOMBI_POLICY_EPS')
t_weight_begin = os.environ.get('BOMBI_WEIGHTS_BEGIN')

# Default values for training.
if t_policy == None: t_policy = 'greedy'
if t_policy_eps == None: t_policy_eps = 0.15
if t_weight_begin == None: t_weight_begin = 'trained'

# Construct unique id for naming of persistent data.
t_training_id = "{}_{}_{}".format(t_policy, t_policy_eps, t_weight_begin)
if t_policy == "greedy":
    t_training_id = "{}_{}".format(t_policy, t_weight_begin)

# Save weights to file
print("TRAINING ID:", t_training_id)
weights_file = "nida_{}_weights.npy".format(t_training_id)



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    # Reward for full game (10 rounds)



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # We need to perform at least one action before computing an
    # intermediary reward.
    if self.game_state['step'] == 1:
        return None

    reward = new_reward(events)
    self.logger.info('Given reward of %s', reward)
    self.accumulated_reward += reward

    # Extract features from the new ("next") game state.
    self.F = RLFeatureExtraction(new_game_state)

    # Get action of the previous step. The previous action was set in
    # the last act() call and is thus named 'next_action'.
    self.prev_action = self_action

    # Keep track of all experiences in the episode for later learning.
    if self.game_state['step'] <= self.replay_buffer_max_steps + 1:
        # The last tuple element defines if there was a transition to
        # a terminal state.
        self.replay_buffer.append((self.F_prev, self.prev_action, reward, self.F, False))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # Add experience for transition to terminal state.
    self.F = RLFeatureExtraction(last_game_state)
    self.prev_action = last_action
    reward = new_reward(events)

    self.replay_buffer.append((self.F_prev, self.prev_action, reward, self.F, True))
    self.logger.info('Given end reward of %s', reward)
    self.accumulated_reward += reward
    self.accumulated_reward_generation += self.accumulated_reward
    self.accumulated_reward = 0
    self.current_round += 1

    # Begin experience replay
    if (self.current_round % self.generation_nrounds) == 0:
        experience_mini_batch = random.sample(self.replay_buffer, self.replay_buffer_sample_size)
        print("REPLAY BUFFER SIZE:", len(self.replay_buffer))
        print("MINI BATCH SIZE:", len(experience_mini_batch))
        self.replay_buffer_sample_size += 20  # set as variable

        # Reset replay buffer
        if self.generation_current % self.replay_buffer_every_ngenerations == 0:
            print("RESETTING REPLAY BUFFER")
            self.replay_buffer = []

        weights_batch_update = np.zeros(len(self.weights))
        for X, A, R, Y, terminal in experience_mini_batch:
            # # Compute maximum Q value and corresponding action.
            # X_A = X.state_action(A)
            # Q_max, A_max = Y.max_q(self.weights)

            # # Do not change the reward if the state is terminal.
            # if terminal == True:
            #     V = R
            # else:
            #     V = R + self.discount * Q_max
            if A != None:
                X_A = X.state_action(A)
                Q_max, A_max = Y.max_q(self.weights)
                TD_error = R + (self.discount * Q_max) - np.dot(X_A, self.weights)

                weights_batch_update = weights_batch_update + self.alpha / self.replay_buffer_sample_size * TD_error * X_A
                # incremental gradient descent
                # self.weights = self.weights + self.alpha / self.replay_buffer_sample_size * TD_error * X_A

        self.weights = self.weights + weights_batch_update
        self.plot_weights.append(self.weights)
        # print("TOTAL REWARD FOR GENERATION {}: {}".format(self.generation_current,
        # self.accumulated_reward_generation))
        # print("WEIGHTS FOR NEXT GENERATION:", self.weights, sep=" ")

        self.generation_current += 1
        self.game_ratio = self.generation_current / self.generation_total

        self.plot_rewards.append(self.accumulated_reward_generation)
        np.save(t_training_id, self.plot_rewards)
        self.accumulated_reward_generation = 0

    np.save(weights_file, self.weights)

def new_reward(events):
    # An action is always punished with a small negative reward due to
    # time constraints on the game.
    reward = 0
    for event in events:
        if event == e.BOMB_DROPPED:
            reward += 1
        elif event == e.COIN_COLLECTED:
            reward += 200
        elif event == e.KILLED_SELF:
            reward -= 1000
        elif event == e.CRATE_DESTROYED:
            reward += 10
        elif event == e.COIN_FOUND:
            reward += 100
        elif event == e.KILLED_OPPONENT:
            reward += 700
        elif event == e.GOT_KILLED:
            reward -= 500
        elif event == e.SURVIVED_ROUND:
            reward += 200
        elif event == e.INVALID_ACTION:
            reward -= 2

    return reward
