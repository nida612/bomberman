import os  # required for training
import random

from agent_code.bombi_agent.feature_extraction import *

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


# weights_file = "./agent_code/nida_agent/models/{}_weights.npy".format(t_training_id)

## REWARDS AND POLICY

def initialize_weights(method):
    best_guess = np.asarray([1, 1.5, -7, -1, 4, -0.5, 1.5, 1, 0.5, 0.8, 0.5, 1, -1, 0.6, 0.6])
    trained_value = np.asarray([1, 2.36889437, -7.40026249, -1.34216494, 3.69025486,
                                -0.64679605, 2.34142438, 1.04329996, 0.50236224,
                                1.25573668, 0.47097264, 1.00000857, -1,
                                0.51097347, 0.22704274])
    if method == 'trained':
        return trained_value
    elif method == 'bestguess':
        return best_guess
    elif method == 'ones':
        return np.ones(len(best_guess))
    elif method == 'zero':
        return np.zeros(len(best_guess))
    elif method == 'random':
        return np.random.rand(1, len(best_guess)).flatten()
    else:
        return None


def policy_select_action(greedy_action, policy, policy_eps, game_ratio, logger=None):
    # Select a random action for use in epsilon-greedy or random-walk
    # exploration.
    random_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'])

    if policy == 'greedy':
        logger.info('Pick greedy action %s', greedy_action)
        return greedy_action
    elif policy == 'epsgreedy':
        logger.info('Picking greedy action at probability %s', 1 - policy_eps)
        return np.random.choice([greedy_action, random_action],
                                p=[1 - policy_eps, policy_eps])
    elif policy == 'diminishing':
        policy_eps_dim = max(0.05, (1 - game_ratio) * policy_eps)
        logger.info('Picking greedy action at probability %s', 1 - policy_eps_dim)
        return np.random.choice([greedy_action, random_action],
                                p=[1 - policy_eps_dim, policy_eps_dim])
    else:
        return None


## GAME METHODS

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    np.random.seed()
    self.accumulated_reward = 0
    self.current_round = 1

    # Hyperparameters for learning methods.
    self.discount = 0.95
    self.alpha = 0.01

    # Values for diminished epsilon policy.
    self.accumulated_reward_generation = 0
    self.generation_current = 1
    self.generation_nrounds = 10
    self.generation_total = max(1, int(10 / self.generation_nrounds))
    self.game_ratio = self.generation_current / self.generation_total

    # Hyperparameters for (prioritized) experience replay.
    self.replay_buffer = []
    self.replay_buffer_max_steps = 200
    self.replay_buffer_update_after_nrounds = 10
    self.replay_buffer_sample_size = 50
    self.replay_buffer_every_ngenerations = 1

    # Load persistent data for exploitation.
    # try:
    #     self.weights = np.load(weights_file)
    #     print("LOADED WEIGHTS", self.weights)
    # except EnvironmentError:
    #     self.weights = initialize_weights(t_weight_begin)
    #     print("INITIALIZED WEIGHTS", self.weights)

    if self.train or not os.path.isfile("nida_greedy_trained_weights.npy"):
        self.logger.info("Setting up model from scratch.")
        self.weights = initialize_weights(t_weight_begin)
        print("INITIALIZED WEIGHTS", self.weights)
    else:
        self.logger.info("Loading model from saved state.")
        self.weights = np.load("nida_greedy_trained_weights.npy")
        print("LOADED WEIGHTS", self.weights)

    # List for storing y values (matplotlib)
    self.plot_rewards = []
    self.plot_weights = [self.weights]


def act(self, blah={}):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """
    # Update features to for the current ("previous") game state.
    self.game_state = blah
    self.F_prev = RLFeatureExtraction(self.game_state)

    # Multiple actions may give the same (optimal) reward. To avoid bias towards
    # a particular action, shuffle the greedy actions.
    Q_max, A_max = self.F_prev.max_q(self.weights)
    random.shuffle(A_max)

    self.next_action = policy_select_action(A_max[0], t_policy, t_policy_eps,
                                            self.game_ratio, self.logger)
    return self.next_action


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
