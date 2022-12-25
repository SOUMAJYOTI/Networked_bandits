import numpy as np
from collections import defaultdict

INFTY = 10.0


def lender_utility(l, b, amount_lenders, borrower_rates):
    return (borrower_rates[b]*amount_lenders[l]) / 1000 #


# Risk preference not considered for now
def borrower_utility(l, b, sim_values):
    return sim_values[b][l]  # 2.0 is for normalization


def get_rewards_list_start(n_l, n_b, u_b, num_sims_per_step, T, variance):
    rewards_from_borrower = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for s_idx in range(num_sims_per_step):
        for t in range(T):
            for l_idx in range(1, n_l+1):
                for b_idx in range(1, n_b+1):
                    sampled_reward = max(0.001, np.random.normal(u_b[b_idx][l_idx], variance, 1)[0])
                    rewards_from_borrower[s_idx][l_idx][b_idx].append(sampled_reward)

    return rewards_from_borrower


def get_rewards_lender(lender_matches, l_idx, u):
    reward_total = 0

    for b_match, frac in lender_matches[l_idx]:
        reward_total += (u[b_match][l_idx])

    return reward_total


def rewards(mean, variance):
    return np.random.normal(mean, variance, 1)[0]


def beta_subgaussian_rewards(mean, variance):
    return np.random.normal(mean, variance, 1)[0]


def lipschitz_functions():
    raise NotImplementedError


def reward_ucb(rewards_list, time):
    if len(rewards_list) == 0:
        return INFTY
    else:
        # print(np.mean(rewards_list), (np.sqrt(1.5 * np.log(time + 1) / len(rewards_list))))
        return np.mean(rewards_list) + (np.sqrt(0.5 * np.log(time + 1) / len(rewards_list)))


def sum_regrets(regret):
    regret_sum_t = defaultdict(lambda: defaultdict(float))

    for l in regret:
        for t in regret[l]:
            for s in range(len(regret[l][t])):
                regret_sum_t[t][s] += regret[l][t][s]

    return regret_sum_t


def max_regrets(regret):
    regret_max_t = defaultdict(lambda: defaultdict(float))

    for l in regret:
        for t in regret[l]:
            for s in range(len(regret[l][t])):
                regret_max_t[t][s] = -10

    for l in regret:
        for t in regret[l]:
            for s in range(len(regret[l][t])):
                if regret[l][t][s] > regret_max_t[t][s]:
                    regret_max_t[t][s] = regret[l][t][s]

    return regret_max_t


def compute_cb_arms(rewards_list, time):
    if len(rewards_list) > 0:
        return np.mean(rewards_list) - (np.sqrt(0.5 * np.log(time + 1) / len(rewards_list))), \
               np.mean(rewards_list) + (np.sqrt(0.5 * np.log(time + 1) / len(rewards_list)))
    else:
        return -INFTY, INFTY


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
        # print(cb_arms[l_idx][borrower_id]['ub'], cb_arms[lender_id][borrower_id]['lb'])
        if cb_arms[l_idx][borrower_id]['ub'] >= 1.5*cb_arms[lender_id][borrower_id]['lb']:
            return False

    # Criterion 2
    for b_idx in borrowers_list:
        if borrower_id == b_idx:
            continue
        # print(cb_arms[lender_id][b_idx]['ub'], cb_arms[lender_id][borrower_id]['lb'])
        if cb_arms[lender_id][b_idx]['ub'] >= 1.5*cb_arms[lender_id][borrower_id]['lb']:
            return False

    # print("----------------------------------------")
    return True


def get_borrower_regret(optimal_lenders, matched_lenders, u_b):
    sum_optimal_u: int = 0
    sum_matched_u: int = 0
    for ol in optimal_lenders:
        sum_optimal_u += u_b[ol]

    for ml in matched_lenders:
        sum_matched_u += u_b[ml]

    return sum_optimal_u - sum_matched_u
