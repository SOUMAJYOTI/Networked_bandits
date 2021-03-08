from collections import defaultdict
from models.lp.gs_matching import *
from utilities.utility_fns import *

VARIANCE = 1.0
NUM_SIMS_PER_STEP = 50


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


def gs_bandit_method_basic(u_b, u_l, c, q, lambda_1, lambda_2, preference_borrowers, preference_lenders):
    T = 40  # horizon, n

    rewards_list_l = defaultdict(lambda: defaultdict(list))
    util_send_l = defaultdict(lambda: defaultdict(float))

    # For each lender l, initialize the current utility for a borrower b as u_l(b)
    # and then initilize the util_send list for matching with the current utility
    for l_idx in q:
        for b_idx in c:
            util_send_l[l_idx][b_idx] = average_reward(u_l[l_idx][b_idx], rewards_list_l[l_idx][b_idx])

    # Baseline GS
    lender_matches_optimal = []
    while len(lender_matches_optimal) < len(q):
        borrower_matches_optimal, lender_matches_optimal, objVal_optimal = gs_bandit_method_baseline(u_b, u_l, c, q,
                                                                                                     lambda_1, lambda_2)

    regret_lender_t = defaultdict(lambda: defaultdict(list))

    for s_idx in range(NUM_SIMS_PER_STEP):
        # print("Simulation no.: " + str(s_idx))
        for t in range(1, T + 1):
            # print("Matching time step " + str(t))
            borrower_matches, lender_matches, objVal = model_gs_matching(u_b, util_send_l, c, q, util_send_l,
                                                                         lambda_1, lambda_2, LogToConsole=False)
            #             if borrower_matches == -1 or lender_matches == -1:
            #                 print("Non-optimal lending")
            #                 for l_idx in range(1, n_l+1):
            #                     regret_lender_t[l_idx].append(-1)
            #                 continue

            for l_idx in q:
                if l_idx in lender_matches:
                    b_match = lender_matches[l_idx]  # matched borrower
                    rewards_list_l[l_idx][b_match].append(
                        rewards(u_l[l_idx][b_match] + u_b[b_match][l_idx],
                                VARIANCE))  # update the reward list for l-b pair
                    util_send_l[l_idx][b_match] = reward_ucb(u_l[l_idx][b_match], rewards_list_l[l_idx][b_match], t)

                    # print(lender_matches_optimal[l_idx], b_match)
                    r = u_l[l_idx][lender_matches_optimal[l_idx]] - u_l[l_idx][b_match]
                else:
                    r = regret_lender_t[l_idx][t - 1][s_idx]

                regret_lender_t[l_idx][t].append(r)

                # print("Lender {} with regret {}".format(l_idx, r))

    return regret_lender_t, borrower_matches, lender_matches, objVal


def gs_bandit_phases(u_b, u_l, c, q, lambda_1, lambda_2):
    T = 10  # horizon, n

    rewards_list_l = defaultdict(lambda: defaultdict(list))
    util_send_l = defaultdict(lambda: defaultdict(float))
    util_send_b = defaultdict(lambda: defaultdict(float))

    # For each lender l, initialize the current utility for a borrower b as u_l(b)
    # and then initilize the util_send list for matching with the current utility
    for l_idx in q:
        for b_idx in c:
            util_send_l[l_idx][b_idx] = reward_ucb(u_l[l_idx][b_idx], [], 0)
            util_send_b[b_idx][l_idx] = u_b[b_idx][l_idx]

    # Baseline GS
    lender_matches_optimal = []
    while len(lender_matches_optimal) < len(q):
        borrower_matches_optimal, lender_matches_optimal, objVal_optimal = gs_bandit_method_baseline(u_b, u_l, c, q,
                                                                                                     lambda_1, lambda_2)

    regret_lender_t = defaultdict(lambda: defaultdict(list))

    for s_idx in range(NUM_SIMS_PER_STEP):
        # print("Simulation no.: " + str(s_idx))
        for t in range(1, T + 1):
            # print("Matching time step " + str(t))
            q_curr = q.copy()
            c_curr = c.copy()
            lenders_unmatched = list(q_curr.keys())
            borrowers_unmatched = list(c_curr.keys())

            # Proceed in phases
            cb_arms = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

            # Initialize the lcb and the ucb for all the arms for each lender
            for l_idx in q:
                for b_idx in c:
                    cb_arms[l_idx][b_idx]['lb'], cb_arms[l_idx][b_idx]['ub'] = compute_cb_arms(u_l[l_idx][b_idx], [], 0)

            # print_cb_arms(cb_arms)
            borrower_matches, lender_matches, objVal = model_gs_matching(util_send_b, util_send_l, c_curr, q_curr,
                                                                         util_send_l,
                                                                         lambda_1, lambda_2, LogToConsole=False)

            if borrower_matches == -1:
                break

            lenders_to_remove = []
            borrowers_to_remove = []

            for b_idx in borrower_matches:
                lenders_b = borrower_matches[b_idx]

                for lb in lenders_b:
                    if is_early_matching_valid(lenders_b, borrowers_unmatched, cb_arms, lb, b_idx):
                        lenders_to_remove.append(lb)
                        c_curr[b_idx] -= q_curr[lb]

                    # update the confidence intervals for b_idx, lb
                    cb_arms[lb][b_idx]['lb'], cb_arms[lb][b_idx]['ub'] = compute_cb_arms(u_l[lb][b_idx],
                                                                                         rewards_list_l[lb][b_idx], t)

                    # update the regret with lb-->b_idx match
                    regret_lender_t[lb][t].append(u_l[lb][lender_matches_optimal[lb]] - u_l[lb][b_idx])

                    # update the utilities for preferences for agents
                    rewards_list_l[lb][b_idx].append(
                        rewards(u_l[lb][b_idx] + u_b[b_idx][lb],
                                VARIANCE))  # update the reward list for l-b pair
                    util_send_l[lb][b_idx] = reward_ucb(u_l[lb][b_idx], rewards_list_l[lb][b_idx], t)

                if c_curr[b_idx] <= 0:
                    # print("Borrower ", b_match, " removed")
                    borrowers_to_remove.append(b_idx)

                for lr in list(set(lenders_to_remove)):
                    lenders_unmatched.remove(lr)

                for br in list(set(borrowers_to_remove)):
                    borrowers_unmatched.remove(br)

                # print(lenders_unmatched)
                # Revise the lender investments dict
                q_curr = revise_agent_dict(q_curr, lenders_unmatched)
                # Revise the borrower requests dict
                c_curr = revise_agent_dict(c_curr, borrowers_unmatched)

                # Revise the borrower and lenders utility dicts
                util_send_l = revise_lb_dict(util_send_l, lenders_unmatched, borrowers_unmatched)
                util_send_b = revise_lb_dict(util_send_b, borrowers_unmatched, lenders_unmatched)

    return regret_lender_t, borrower_matches, lender_matches, objVal
