import numpy as np
from collections import defaultdict

INFTY = 1.0

def lender_utility(l, b, sim_values, amount_lenders, borrower_rates):
    print((borrower_rates[b]*amount_lenders[l]) / 30)
    return (borrower_rates[b]*amount_lenders[l]) / 30 #sim_values[b][l] #

# Risk preference not considered for now
def borrower_utility(l, b, sim_values, risk_preference):
    return (sim_values[b][l] + risk_preference[b][l]) / 2.0  # 2.0 is for normalization

def get_rewards_list_start(n_l, n_b, u_b, num_sims_per_step, T, variance):
    rewards_from_borrower = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for s_idx in range(num_sims_per_step):
        for t in range(T):
            for l_idx in range(1, n_l+1):
                for b_idx in range(1, n_b+1):
                    rewards_from_borrower[s_idx][l_idx][b_idx].append(np.random.normal(u_b[b_idx][l_idx], variance, 1)[0])

    return rewards_from_borrower

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


def revise_util_dict(orig_dict, agents_from, agents_to):
    new_dict = defaultdict(lambda: defaultdict(float))
    for af in agents_from:
        for at in agents_to:
            new_dict[af][at] = orig_dict[af][at]
    return new_dict


def revise_agent_dict(orig_dict, agents):
    new_dict = defaultdict(float)
    for a in agents:
        new_dict[a] = orig_dict[a]
    return new_dict


def print_cb_arms(cb_arms):
    for l_idx in cb_arms:
        for b_idx in cb_arms[l_idx]:
            print(l_idx, b_idx, cb_arms[l_idx][b_idx]['lb'], cb_arms[l_idx][b_idx]['ub'])


def is_early_matching_valid(lenders_list, borrowers_list, cb_arms, lender_id, borrower_id):
    # Criterion 1
    for l_idx in lenders_list:
        if l_idx == lender_id:
            continue
        if cb_arms[l_idx][borrower_id]['ub'] >= cb_arms[lender_id][borrower_id]['lb']:
            return False

    # Criterion 2
    for b_idx in borrowers_list:
        if borrower_id == b_idx:
            continue
        if cb_arms[lender_id][b_idx]['ub'] >= cb_arms[lender_id][borrower_id]['lb']:
            return False

    return True
