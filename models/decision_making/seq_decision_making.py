from collections import defaultdict
from models.lp.gs_matching import *
from utilities.utility_fns import *

VARIANCE = 0.3
NUM_SIMS_PER_STEP = 50


def gs_bandit_method_baseline(u_b, u_l, c, q, lambda_1, lambda_2):
    u = defaultdict(lambda: defaultdict(float))
    obj = defaultdict(lambda: defaultdict(float))
    for l_idx in q:
        for b_idx in c:
            # we optimize for lender_borrower returns
            u[l_idx][b_idx] = u_b[b_idx][l_idx] + u_l[l_idx][b_idx]  # lender utility + borrower utility
            obj[b_idx][l_idx] = u[l_idx][b_idx]  # objective fn utility same as lender-borrower utility

    borrower_matches_optimal, lender_matches_optimal, objVal = model_gs_matching(u_b, u, c, q, obj, lambda_1,
                                                                                 lambda_2, LogToConsole=False)

    return lender_matches_optimal, objVal


def gs_bandit_method_basic(u_b, u_l, c, q, lambda_1, lambda_2, preference_borrowers, preference_lenders):
    T = 20  # horizon, n

    rewards_list_l = defaultdict(lambda: defaultdict(list))
    util_send_l = defaultdict(lambda: defaultdict(float))

    # For each lender l, initialize the current utility for a borrower b as u_l(b)
    # and then initilize the util_send list for matching with the current utility
    for l_idx in q:
        for b_idx in c:
            util_send_l[l_idx][b_idx] = u_l[l_idx][b_idx]

    # Baseline GS
    lender_matches_optimal = []
    while (len(lender_matches_optimal) < len(q)):
        lender_matches_optimal, objVal_optimal = gs_bandit_method_baseline(u_b, u_l, c, q, lambda_1, lambda_2)

    regret_lender_t = defaultdict(lambda: defaultdict(list))

    for s_idx in range(NUM_SIMS_PER_STEP):
        print("Simulation no.: " + str(s_idx))
        for t in range(1, T + 1):
            print("Matching time step " + str(t))
            borrower_matches, lender_matches, objVal = model_gs_matching(util_send_l, c, q, util_send_l,
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
                        rewards(u_b[b_match][l_idx], VARIANCE))  # update the reward list for l-b pair
                    util_send_l[l_idx][b_match] = reward_ucb(u_l[l_idx][b_match], rewards_list_l[l_idx][b_match], t)

                    # print(lender_matches_optimal[l_idx], b_match)
                    r = u_l[l_idx][lender_matches_optimal[l_idx]] - u_l[l_idx][b_match]
                else:
                    r = regret_lender_t[l_idx][t - 1][s_idx]

                regret_lender_t[l_idx][t].append(r)

                # print("Lender {} with regret {}".format(l_idx, r))

    return regret_lender_t
