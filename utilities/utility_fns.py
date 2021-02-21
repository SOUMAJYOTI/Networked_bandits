import numpy as np

INFTY = 10

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
        return average_reward(base_reward, rewards_list) + np.sqrt(1.5 * np.log(time + 1) / len(rewards_list))


def cb_arms(base_reward, rewards_list, time):
    if len(rewards_list) > 0:
        return average_reward(base_reward, rewards_list) - np.sqrt(1.5 * np.log(time + 1) / len(rewards_list)), \
               average_reward(base_reward, rewards_list) + np.sqrt(1.5 * np.log(time + 1) / len(rewards_list))
    else:
        return average_reward(base_reward, rewards_list) - INFTY, average_reward(base_reward, rewards_list) + INFTY


# def find_interesction_arms(lcb_m, ucb_m, interval_list_arms, b):
#     intersecting_arms = []
#     for b_idx in interval_list_arms:
#         if b == b_idx:
#             continue
#         lcb, ucb = interval_list_arms[b_idx]
#         if lcb < ucb_m or ucb > lcb_m:
#             intersecting_arms.append(l_idx)
#
#     return intersecting_arms
