from simulations.simulation import *
from utilities.plots import *
import os

VARIANCE = 1
NUM_SIMS_PER_STEP = 50
T = 50  # horizon, n

if __name__ == "__main__":
    borrower_matches_optimal = -1
    while borrower_matches_optimal == -1:
        u_b, u_l, c, q, lambda_1, lambda_2, preference_borrowers, preference_lenders = simulate_lending()
        rewards_from_borrowers = get_rewards_list_start(len(u_l), len(u_b), u_b, NUM_SIMS_PER_STEP, T, VARIANCE)

        borrower_matches_optimal, lender_matches_optimal, objVal = gs_bandit_method_baseline(u_b, u_l, c, q, lambda_1,
                                                                                             lambda_2)  # baseline

    # print("-----------------------------------Optimal matching-----------------------------------")
    # for b_idx in borrower_matches_optimal:
    #     sum_l = 0
    #     print("Borrower ", b_idx, " with request ", c[b_idx])
    #     for l_idx in borrower_matches_optimal[b_idx]:
    #         print("Lender", l_idx, " invested ", q[l_idx])
    #         sum_l += q[l_idx]
    #     print("Total amount raised by borrower ", b_idx, " is: ", sum_l)
    #
    # print("-----------------------------------GS Basic-----------------------------------")
    regret_lenders_basic, borrower_matches, lender_matches, objVal = gs_bandit_method_basic(u_b, u_l, c, q, lambda_1,
                                                                                      lambda_2, preference_borrowers,
                                                                                      preference_lenders, NUM_SIMS_PER_STEP, T, rewards_from_borrowers, VARIANCE)
    # for b_idx in borrower_matches:
    #     sum_l = 0
    #     print("Borrower ", b_idx, " with request ", c[b_idx])
    #     for l_idx in borrower_matches[b_idx]:
    #         print("Lender", l_idx, " invested ", q[l_idx])
    #         sum_l += q[l_idx]
    #     print("Total amount raised by borrower ", b_idx, " is: ", sum_l)

    plot_lines(regret_lenders_basic, save_dir="../figures/figure_3_07/10b_60l/basic")

    print("-----------------------------------GS BLEMET -----------------------------------")
    regret_lenders_blemet, borrower_matches, lender_matches, objVal = gs_bandit_BLEMET(u_b, u_l, c, q, lambda_1,
                                                                                      lambda_2,  NUM_SIMS_PER_STEP, T, rewards_from_borrowers, VARIANCE)
    # for b_idx in borrower_matches:
    #     sum_l = 0
    #     print("Borrower ", b_idx, " with request ", c[b_idx])
    #     for l_idx in borrower_matches[b_idx]:
    #         print("Lender", l_idx, " invested ", q[l_idx])
    #         sum_l += q[l_idx]
    #     print("Total amount raised by borrower ", b_idx, " is: ", sum_l)

    plot_lines_blemet(regret_lenders_blemet, save_dir="../figures/figure_3_07/10b_60l/phases/")
    # #
    plot_lines_aggregate(regret_lenders_basic, regret_lenders_blemet, save_dir="../figures/figure_3_07/10b_60l/agg/")
    #
