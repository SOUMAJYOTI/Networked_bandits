from collections import defaultdict
from models.lp.gs_matching import *
from utilities.utility_fns import *
import copy


def gs_bandit_method_baseline(u_b, u_l, c, q, lambda_1, lambda_2):
    u = defaultdict(lambda: defaultdict(float))
    obj = defaultdict(lambda: defaultdict(float))
    for l_idx in q:
        for b_idx in c:
            # we optimize for lender_borrower returns
            u[l_idx][b_idx] = u_b[b_idx][l_idx] + u_l[l_idx][b_idx]  # lender utility + borrower utility
            obj[b_idx][l_idx] = u[l_idx][b_idx]  # objective fn utility same as lender-borrower utility

    try:
        borrower_matches_optimal, lender_matches_optimal, objVal = model_gs_matching(u_b, u, c, q, obj, lambda_1,
                                                                                     lambda_2, LogToConsole=False)
    except:
        print("************optimal Soln. not found, trying another configuration.......************")
        borrower_matches_optimal, lender_matches_optimal, objVal = -1, -1, -1

    return borrower_matches_optimal, lender_matches_optimal, objVal


def gs_bandit_method_basic(u_b, u_l, c, q, lambda_1, lambda_2, preference_borrowers, preference_lenders,
                           NUM_SIMS_PER_STEP, T, rewards_from_borrower, VARIANCE):
    util_send_l_orig = defaultdict(lambda: defaultdict(float))

    # For each lender l, initialize the current utility for a borrower b as u_l(b)
    # and then initilize the util_send list for matching with the current utility
    for l_idx in q:
        for b_idx in c:
            util_send_l_orig[l_idx][b_idx] = u_l[l_idx][b_idx]

    # Baseline GS
    lender_matches_optimal = []
    while len(lender_matches_optimal) < len(q):
        borrower_matches_optimal, lender_matches_optimal, objVal_optimal = gs_bandit_method_baseline(u_b, u_l, c, q,
                                                                                                     lambda_1, lambda_2)
    regret_lender_t = defaultdict(lambda: defaultdict(list))
    ub_t = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for s_idx in range(NUM_SIMS_PER_STEP):
        print("Simulation no.: " + str(s_idx))

        util_send_l = copy.deepcopy(util_send_l_orig)
        rewards_list_l = defaultdict(lambda: defaultdict(list))

        sum_rewards = defaultdict(list)

        obj = defaultdict(lambda: defaultdict(float))
        for l_idx in q:
            for b_idx in c:
                # we optimize for lender_borrower returns
                util_send_l[l_idx][b_idx] = reward_ucb(rewards_list_l[l_idx][b_idx], 1)
                obj[b_idx][l_idx] = util_send_l[l_idx][b_idx]  # lender utility

        for t in range(1, T + 1):
            # print("Matching time step " + str(t))
            borrower_matches, lender_matches, objVal = model_gs_matching(u_b, util_send_l, c, q, obj,
                                                                         lambda_1, lambda_2, LogToConsole=False)

            if borrower_matches == -1:
                for l_idx in q:
                    r = regret_lender_t[l_idx][t - 1][s_idx]
                    regret_lender_t[l_idx][t].append(r)
                continue

            for l_idx in q:
                if l_idx in lender_matches:
                    b_match = lender_matches[l_idx]  # matched borrower
                    rewards_list_l[l_idx][b_match].append(
                        rewards_from_borrower[s_idx][l_idx][b_match][t - 1])  # update the reward list for l-b pair
                    util_send_l[l_idx][b_match] = reward_ucb(rewards_list_l[l_idx][b_match], t)

                    sum_rewards[l_idx].append(u_l[l_idx][b_match] + u_b[b_match][l_idx])

                    # print(lender_matches_optimal[l_idx], b_match)
                    optimal_borrower = lender_matches_optimal[l_idx]
                    # print(t, util_send_l[l_idx][b_match], l_idx, optimal_borrower, b_match)

                    r = t * (u_l[l_idx][optimal_borrower] + u_b[optimal_borrower][l_idx]) - \
                        np.sum(sum_rewards[l_idx])
                    # if l_idx == 10:
                    #     print(s_idx, t, r, b_match, optimal_borrower)
                    # print(l_idx, np.mean(rewards_list_l[l_idx][b_match]), util_send_l[l_idx][b_match])
                    obj[b_match][l_idx] = np.mean(rewards_list_l[l_idx][b_match]) + util_send_l[l_idx][b_match]
                    # ub_t[l_idx][b_match][t].append(compute_cb_arms(u_l[l_idx][b_match], rewards_list_l[l_idx][b_match])[1])
                else:
                    r = regret_lender_t[l_idx][t - 1][s_idx]

                regret_lender_t[l_idx][t].append(r)

            # for b in c:
            #     print(b, compute_cb_arms(rewards_list_l[5][b], t))
            # print("---------------------------------------------------")

            # print("Lender {} with regret {}".format(l_idx, r))

    return regret_lender_t, borrower_matches, lender_matches, objVal


def gs_bandit_BLEMET(u_b, u_l, c, q, lambda_1, lambda_2, lambda_3,
                     NUM_SIMS_PER_STEP, T, rewards_from_borrower, with_fairness=False, VARIANCE=1.0):
    util_send_l_orig = defaultdict(lambda: defaultdict(float))
    util_send_b_orig = defaultdict(lambda: defaultdict(float))

    borrower_final_matches = defaultdict(list)
    lender_final_matches = defaultdict(int)

    # For each lender l, initialize the current utility for a borrower b as u_l(b)
    # and then initilize the util_send list for matching with the current utility
    for l_idx in q:
        for b_idx in c:
            util_send_l_orig[l_idx][b_idx] = reward_ucb(u_l[l_idx][b_idx], [], 0)
            util_send_b_orig[b_idx][l_idx] = u_b[b_idx][l_idx]

    # Baseline GS
    lender_matches_optimal = []
    while len(lender_matches_optimal) < len(q):
        borrower_matches_optimal, lender_matches_optimal, objVal_optimal = gs_bandit_method_baseline(u_b, u_l, c, q,
                                                                                                     lambda_1, lambda_2)

    regret_lender_t = defaultdict(lambda: defaultdict(list))

    for s_idx in range(NUM_SIMS_PER_STEP):
        print("Simulation no.: " + str(s_idx))
        q_curr = copy.deepcopy(q)
        c_curr = copy.deepcopy(c)

        u_l_curr = copy.deepcopy(u_l)
        util_send_l = copy.deepcopy(util_send_l_orig)
        util_send_b = copy.deepcopy(util_send_b_orig)
        rewards_list_l = defaultdict(lambda: defaultdict(list))

        lenders_unmatched = list(q_curr.keys())
        borrowers_unmatched = list(c_curr.keys())

        cb_arms = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        obj = defaultdict(lambda: defaultdict(float))
        # Initialize the lcb and the ucb for all the arms for each lender
        for l_idx in q:
            for b_idx in c:
                cb_arms[l_idx][b_idx]['lb'], cb_arms[l_idx][b_idx]['ub'] = compute_cb_arms(u_l[l_idx][b_idx], [], 0)

        # for lb in u_l:
        #     max_ub = -1
        #     for b_idx in u_b:
        #         if cb_arms[lb][b_idx]['ub'] > max_ub:
        #             max_ub = cb_arms[lb][b_idx]['ub']
        #             bstart = b_idx
        #
        #     regret_lender_t[lb][0].append(u_l[lb][lender_matches_optimal[lb]] - u_l[lb][bstart])

        for t in range(1, T + 1):
            # print("Matching time step " + str(t))
            # print(t, lenders_unmatched, borrowers_unmatched)
            if len(lenders_unmatched) == 0 or len(borrowers_unmatched) == 0:
                break

            # Find the maximum remaining amount among all borrowers
            b_rem_max = -10
            for b_idx in borrowers_unmatched:
                if c_curr[b_idx] > b_rem_max:
                    b_rem_max = c_curr[b_idx]

            fair_constr = b_rem_max * np.exp((t / T))
            if with_fairness:
                borrower_matches, lender_matches, objVal = model_gs_matching_with_fairness(util_send_b, util_send_l,
                                                                                           c_curr, q_curr,
                                                                                           obj, fair_constr,
                                                                                           lambda_1, lambda_2, lambda_3,
                                                                                           LogToConsole=False)
            else:
                borrower_matches, lender_matches, objVal = model_gs_matching(util_send_b, util_send_l, c_curr, q_curr,
                                                                             obj,
                                                                             lambda_1, lambda_2, LogToConsole=False)
            if borrower_matches == -1:
                break

            lenders_to_remove = []
            borrowers_to_remove = []

            for b_idx in borrower_matches:
                lenders_b = borrower_matches[b_idx]  # lender matches list for b_idx

                for lb in lenders_b:
                    if is_early_matching_valid(lenders_b, borrowers_unmatched, cb_arms, lb, b_idx):
                        lenders_to_remove.append(lb)
                        c_curr[b_idx] -= q_curr[lb]

                    # update the regret with lb-->b_idx match
                    regret_lender_t[lb][t].append(u_l[lb][lender_matches_optimal[lb]] - u_l[lb][b_idx])

                    # update the utilities for preferences for agents
                    rewards_list_l[lb][b_idx].append(
                        rewards_from_borrower[s_idx][lb][b_idx][t - 1])  # update the reward list for l-b pair

                    obj[b_idx][lb] = np.mean(rewards_list_l[lb][b_idx]) + util_send_l[lb][b_idx]

                    # update the utilities for next round
                    # util_send_l[lb][b_idx] = reward_ucb(util_send_l[lb][b_idx], rewards_list_l[lb][b_idx], t)
                    #
                    # # update the confidence intervals for b_idx, lb
                    # cb_arms[lb][b_idx]['lb'], cb_arms[lb][b_idx]['ub'] = compute_cb_arms(util_send_l[lb][b_idx],
                    #                                                                            rewards_list_l[lb][b_idx], t)

                if c_curr[b_idx] <= 0:
                    # print("Borrower ", b_match, " removed")
                    borrowers_to_remove.append(b_idx)

            for lr in list(set(lenders_to_remove)):
                # print(t, lr)
                bm = lender_matches[lr]
                borrower_final_matches[bm].append(lr)
                lender_final_matches[lr] = bm
                lenders_unmatched.remove(lr)

                # fill the remaining t's as same regret as now for lr
                for t_left in range(t + 1, T):
                    regret_lender_t[lr][t_left].append(regret_lender_t[lr][t][-1])

            for br in list(set(borrowers_to_remove)):
                borrowers_unmatched.remove(br)

            # Revise the lender investments dict
            q_curr = revise_agent_dict(q_curr, lenders_unmatched)
            # Revise the borrower requests dict
            c_curr = revise_agent_dict(c_curr, borrowers_unmatched)

            # revise the estimates of the lender utility based on current market situation
            for l_idx in lenders_unmatched:
                for b_idx in borrowers_unmatched:
                    diff = np.max([c_curr[b_idx] - q_curr[l_idx], 0.001])
                    # diff = c_curr[b_idx] - q_curr[l_idx]
                    chance = 1.0 - np.exp(-0.5 * diff)
                    u_l_curr[l_idx][b_idx] = u_l[l_idx][b_idx] * chance

                    # update the utilities for next round
                    util_send_l[l_idx][b_idx] = reward_ucb(rewards_list_l[l_idx][b_idx], t)

                    # update the confidence intervals for b_idx, lb
                    cb_arms[l_idx][b_idx]['lb'], cb_arms[l_idx][b_idx]['ub'] = compute_cb_arms(
                        rewards_list_l[l_idx][b_idx], t)

            # Revise the borrower and lenders utility dicts
            util_send_l = revise_util_dict(util_send_l, lenders_unmatched, borrowers_unmatched)
            util_send_b = revise_util_dict(util_send_b, borrowers_unmatched, lenders_unmatched)
            obj = revise_util_dict(obj, borrowers_unmatched, lenders_unmatched)

    return regret_lender_t, borrower_final_matches, lender_final_matches, objVal
