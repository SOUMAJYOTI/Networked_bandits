from collections import defaultdict
from models.lp.gs_matching import *
from utilities.utility_fns import *
import copy
import dill


def gs_bandit_method_baseline(u_b, u_l, c, q, lambda_1, lambda_2):
    u = defaultdict(lambda: defaultdict(float))
    obj = defaultdict(lambda: defaultdict(float))
    for l_idx in q:
        for b_idx in c:
            # we optimize for lender_borrower returns
            u[l_idx][b_idx] = u_b[b_idx][l_idx] + u_l[l_idx][b_idx]  # lender utility + borrower utility
            obj[b_idx][l_idx] = u[l_idx][b_idx]  # objective fn utility same as lender-borrower utility

    try:
        borrower_matches_optimal, lender_matches_optimal, objVal = model_gs_matching_manytomany(u_b, u, c, q, obj, lambda_1,
                                                                                     lambda_2, LogToConsole=False)
    except:
        print("************optimal Soln. not found, trying another configuration.......************")
        borrower_matches_optimal, lender_matches_optimal, objVal = -1, -1, -1

    print( borrower_matches_optimal, lender_matches_optimal)

    return borrower_matches_optimal, lender_matches_optimal, objVal


def gs_bandit_method_basic(u_b, u_l, c, q, lambda_1, lambda_2, preference_borrowers, preference_lenders,
                           NUM_SIMS_PER_STEP, T, rewards_from_borrower, VARIANCE, objective_type):
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
    regret_borrower_t = defaultdict(lambda: defaultdict(list))

    arms_ts = {}
    for l_idx in q:
        arms_ts[l_idx] = np.zeros((len(c), T))

    arms_ts_mostpref = {}
    for l_idx in q:
        arms_ts_mostpref[l_idx] = np.zeros((len(c), T))

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
                if objective_type == "lenders":
                    obj[b_idx][l_idx] = util_send_l[l_idx][b_idx]  # lender utility
                else:
                    obj[b_idx][l_idx] = u_b[b_idx][l_idx]  # borrower utility

        for t in range(1, T + 1):
            # print("Matching time step " + str(t))
            borrower_matches, lender_matches, objVal = model_gs_matching_manytomany(u_b, util_send_l, c, q, obj,
                                                                         lambda_1, lambda_2, LogToConsole=False)
            if borrower_matches == -1:
                print("No optimal soln. found !!")
                for l_idx in q:
                    r = regret_lender_t[l_idx][t - 1][s_idx]
                    regret_lender_t[l_idx][t].append(r)

                continue

            for l_idx in q:
                if l_idx in lender_matches:
                    b_matches_list = lender_matches[l_idx]  # matched borrowers

                    # find the most preferred borrower for lender l_idx in this step
                    utilmax = -10
                    for b_l in util_send_l[l_idx]:
                        if util_send_l[l_idx][b_l] > utilmax:
                            utilmax = util_send_l[l_idx][b_l]
                            lender_mostpref = b_l

                    reward_total = 0
                    for b_match, frac in b_matches_list:
                        arms_ts[l_idx][b_match-1][t-1] += 1

                        if b_match == lender_mostpref:
                            arms_ts_mostpref[l_idx][b_match-1][t-1] += 1

                        reward_frac = frac * rewards_from_borrower[s_idx][l_idx][b_match][t-1]
                        rewards_list_l[l_idx][b_match].append(
                            reward_frac)  # update the reward list for l-b pair
                        util_send_l[l_idx][b_match] = reward_ucb(rewards_list_l[l_idx][b_match], t)
                        reward_total += (frac * u_l[l_idx][b_match])
                        if objective_type == "lenders":
                            obj[b_match][l_idx] = np.mean(rewards_list_l[l_idx][b_match]) + util_send_l[l_idx][b_match]

                    sum_rewards[l_idx].append(reward_total)

                    optimal_util_l = get_rewards_lender(lender_matches_optimal, u_l)
                    r = t*optimal_util_l - np.sum(sum_rewards[l_idx])
                else:
                    r = regret_lender_t[l_idx][t - 1][s_idx]

                regret_lender_t[l_idx][t].append(r)

    return regret_lender_t, arms_ts, arms_ts_mostpref, borrower_matches, lender_matches, objVal


def gs_bandit_BLEMET(u_b, u_l, c, q, lambda_1, lambda_2, lambda_3,
                     NUM_SIMS_PER_STEP, T, rewards_from_borrower, with_fairness, VARIANCE, objective_type='lenders'):

    util_send_l_orig = defaultdict(lambda: defaultdict(float))
    util_send_b_orig = defaultdict(lambda: defaultdict(float))

    borrower_final_matches = defaultdict(list)
    lender_final_matches = defaultdict(int)

    # For each lender l, initialize the current utility for a borrower b as u_l(b)
    # and then initilize the util_send list for matching with the current utility
    for l_idx in q:
        for b_idx in c:
            util_send_l_orig[l_idx][b_idx] = u_l[l_idx][b_idx]
            util_send_b_orig[b_idx][l_idx] = u_b[b_idx][l_idx]

    # Baseline GS
    lender_matches_optimal = []
    while len(lender_matches_optimal) < len(q):
        borrower_matches_optimal, lender_matches_optimal, objVal_optimal = gs_bandit_method_baseline(u_b, u_l, c, q,
                                                                                                     lambda_1, lambda_2)

    regret_lender_t = defaultdict(lambda: defaultdict(list))
    regret_borrower_t = defaultdict(lambda: defaultdict(list))

    arms_ts = {}
    for l_idx in q:
        arms_ts[l_idx] = np.zeros((len(c), T))

    for s_idx in range(NUM_SIMS_PER_STEP):
        print("Simulation no.: " + str(s_idx))
        q_curr = copy.deepcopy(q)
        c_curr = copy.deepcopy(c)

        util_send_l = copy.deepcopy(util_send_l_orig)
        util_send_b = copy.deepcopy(util_send_b_orig)

        rewards_list_l = defaultdict(lambda: defaultdict(list))

        lenders_unmatched = list(q_curr.keys())
        borrowers_unmatched = list(c_curr.keys())
        sum_rewards = defaultdict(list)
        sum_rewards_borrower = defaultdict(list)

        cb_arms = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        obj = defaultdict(lambda: defaultdict(float))
        # Initialize the lcb and the ucb for all the arms for each lender
        for l_idx in q:
            for b_idx in c:
                # we optimize for lender_borrower returns
                util_send_l[l_idx][b_idx] = reward_ucb(rewards_list_l[l_idx][b_idx], 1)
                if objective_type == "lenders":
                    obj[b_idx][l_idx] = util_send_l[l_idx][b_idx]  # lender utility
                else:
                    obj[b_idx][l_idx] = u_b[b_idx][l_idx]  # borrower utility
                cb_arms[l_idx][b_idx]['lb'], cb_arms[l_idx][b_idx]['ub'] = compute_cb_arms([], 0)

        for t in range(1, T + 1):
            # print("Matching time step " + str(t))
            if len(lenders_unmatched) == 0 or len(borrowers_unmatched) == 0:
                break

            # Find the maximum remaining amount among all borrowers
            b_rem_max = -10
            for b_idx in borrowers_unmatched:
                if (c_curr[b_idx]/c[b_idx]) > b_rem_max:
                    b_rem_max = (c_curr[b_idx]/c[b_idx])

            fair_constr = b_rem_max * np.exp((t/T))
            if with_fairness:
                borrower_matches, lender_matches, objVal = model_gs_matching_with_fairness(u_b, util_send_l,
                                                                                           c_curr, q_curr,
                                                                                           obj, fair_constr,
                                                                                           lambda_1, lambda_2, lambda_3,
                                                                                           LogToConsole=False)
            else:
                borrower_matches, lender_matches, objVal = model_gs_matching(u_b, util_send_l, c_curr, q_curr,
                                                                             obj,
                                                                             lambda_1, lambda_2,  LogToConsole=False)
            lenders_to_remove = []
            borrowers_to_remove = []

            if borrower_matches == -1:
                print("No optimal soln. found !!")
                for l_idx in q:
                    r = regret_lender_t[l_idx][t - 1][s_idx]
                    regret_lender_t[l_idx][t].append(r)

                continue

            for b_idx in borrower_matches:
                lenders_b = borrower_matches[b_idx] # lender matches list for b_idx
                sum_matched_u: int = 0
                for ml in lenders_b:
                    sum_matched_u += u_b[b_idx][ml]

                sum_optimal_u: int = 0
                for ol in borrower_matches_optimal[b_idx]:
                    sum_optimal_u += u_b[b_idx][ol]

                sum_rewards_borrower[b_idx].append(sum_matched_u)
                r_borrower = t * (sum_optimal_u) - np.sum(sum_rewards_borrower[b_idx])
                regret_borrower_t[b_idx][t].append(r_borrower)

                for lb in lenders_b:
                    arms_ts[lb][b_idx-1][t-1] += 1
                    if is_early_matching_valid(lenders_b, borrowers_unmatched, cb_arms, lb, b_idx):
                        lenders_to_remove.append(lb)
                        c_curr[b_idx] -= q_curr[lb]

                    rewards_list_l[lb][b_idx].append(
                        rewards_from_borrower[s_idx][lb][b_idx][t - 1])  # update the reward list for l-b pair
                    util_send_l[lb][b_idx] = reward_ucb(rewards_list_l[lb][b_idx], t)

                    sum_rewards[lb].append(u_l[lb][b_idx])

                    optimal_borrower = lender_matches_optimal[lb]

                    r = t * (u_l[lb][optimal_borrower]) - np.sum(sum_rewards[lb])
                    regret_lender_t[lb][t].append(r)

                    if objective_type == "lenders":
                        obj[b_idx][lb] = np.mean(rewards_list_l[lb][b_idx]) + util_send_l[lb][b_idx]

                    # update the confidence intervals for b_idx, lb
                    cb_arms[lb][b_idx]['lb'], cb_arms[lb][b_idx]['ub'] = compute_cb_arms(
                        rewards_list_l[lb][b_idx], t)

                if c_curr[b_idx] <= 0:
                    borrowers_to_remove.append(b_idx)

            # ------------------------------------------------------------------------------
            # EARLY TERMINATION
            for lr in list(set(lenders_to_remove)):
                # print(t, lr)
                lenders_unmatched.remove(lr)

                # fill the remaining t's as same regret as now for lr
                for t_left in range(t + 1, T):
                    regret_lender_t[lr][t_left].append(regret_lender_t[lr][t][-1])

            for br in list(set(borrowers_to_remove)):
                borrowers_unmatched.remove(br)

                # fill the remaining t's as same regret as now for br
                for t_left in range(t + 1, T):
                    regret_borrower_t[br][t_left].append(regret_borrower_t[br][t][-1])

            # Revise the lender investments dict
            q_curr = revise_agent_dict(q_curr, lenders_unmatched)
            # Revise the borrower requests dict
            c_curr = revise_agent_dict(c_curr, borrowers_unmatched)

            # Revise the borrower and lenders utility dicts
            util_send_l = revise_util_dict(util_send_l, lenders_unmatched, borrowers_unmatched)
            util_send_b = revise_util_dict(util_send_b, borrowers_unmatched, lenders_unmatched)
            obj = revise_util_dict(obj, borrowers_unmatched, lenders_unmatched)

    return regret_lender_t, regret_borrower_t, arms_ts, borrower_final_matches, lender_final_matches, objVal
