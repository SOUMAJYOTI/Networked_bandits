import numpy as np
from collections import defaultdict

INFTY = 0.5


def lender_utility(l, b, sim_values, amount_lenders, borrower_rates):
    return sim_values[b][l]  # + borrower_rates[b]*amount_lenders[l]


# Risk preference not considered for now
def borrower_utility(l, b, sim_values, risk_preference):
    return (sim_values[b][l] + risk_preference[b][l]) / 2.0  # 2.0 is for normalization


def rewards(mean, variance):
    return np.random.normal(mean, variance, 1)[0]


def average_reward(base_reward, rewards_list):
    if len(rewards_list) == 0:
        return base_reward
    else:
        return (base_reward + np.sum(rewards_list)) / (len(rewards_list) + 1)


def reward_ucb(base_reward, rewards_list, time):
    if len(rewards_list) == 0:
        return average_reward(base_reward, rewards_list) + INFTY
    else:
        return average_reward(base_reward, rewards_list) + (np.sqrt(1.5 * np.log(time + 1) / len(rewards_list)))


def compute_cb_arms(base_reward, rewards_list, time):
    if len(rewards_list) > 0:
        return average_reward(base_reward, rewards_list) - (np.sqrt(1.5 * np.log(time + 1) / len(rewards_list))), \
               average_reward(base_reward, rewards_list) + (np.sqrt(1.5 * np.log(time + 1) / len(rewards_list)))
    else:
        return average_reward(base_reward, rewards_list) - INFTY, average_reward(base_reward, rewards_list) + INFTY


def find_interescting_arms(intervals_arms, b):
    intersecting_arms = []
    lcb_m, ucb_m = intervals_arms[b]['lb'], intervals_arms[b]['ub']
    for b_idx in intervals_arms:
        if b == b_idx:
            continue
        lcb, ucb = intervals_arms[b_idx]['lb'], intervals_arms[b_idx]['ub']
        if lcb < ucb_m and ucb > lcb_m:
            intersecting_arms.append(b_idx)

    return intersecting_arms


def equal_distribution_utility():
    raise NotImplementedError("Utilities not implemented")


def revise_lb_dict(orig_dict, agents_from, agents_to):
    new_dict = defaultdict(lambda: defaultdict(float))
    for af in agents_from:
        for at in agents_to:
            new_dict[af][at] = orig_dict[af][at]


def revise_agent_dict(orig_dict, agents):
    new_dict = defaultdict(float)
    for a in agents:
        new_dict[a] = orig_dict[a]
    return new_dict


def print_cb_arms(cb_arms):
    for l_idx in cb_arms:
        for b_idx in cb_arms[l_idx]:
            print(l_idx, b_idx, cb_arms[l_idx][b_idx]['lb'], cb_arms[l_idx][b_idx]['ub'])
