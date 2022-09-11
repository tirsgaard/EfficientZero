import numpy as np

import cytree as tree
import cytree

num = 5
value_delta_max = 0.01
lstm_horizon_len = 5
num_simulations = 10


pb_c_base = 19652
pb_c_init = 1.25
discount = 0.997
action_space_size = (4,)
env_nums = 5
root_dirichlet_alpha = 0.3
root_exploration_fraction = 0.25
value_prefix_pool = [0, 1, 2, 3, 4]
policy_logits_pool = [0, 1, 2, 3, 4]

roots = cytree.Roots(env_nums, action_space_size, num_simulations)
noises = [np.random.dirichlet([root_dirichlet_alpha] * action_space_size).astype(np.float32).tolist() for _ in range(env_nums)]
roots.prepare(root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)


min_max_stats_lst = tree.MinMaxStatsList(num)
min_max_stats_lst.set_delta(value_delta_max)
horizons = lstm_horizon_len

for index_simulation in range(num_simulations):
    hidden_states = []
    hidden_states_c_reward = []
    hidden_states_h_reward = []

    # prepare a result wrapper to transport results between python and c++ parts
    results = tree.ResultsWrapper(num)
    # traverse to select actions for each root
    # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
    # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
    # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
    hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.batch_traverse(roots, pb_c_base, pb_c_init,
                                                                                           discount, min_max_stats_lst,
                                                                                           results)