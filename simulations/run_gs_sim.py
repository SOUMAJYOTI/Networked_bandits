from simulations.simulation import *
from utilities.plots import *
import os
import pickle

VARIANCE = 1
NUM_SIMS_PER_STEP = 50
T = 100 # horizon, n

if __name__ == "__main__":
    borrower_matches_optimal = -1
    while borrower_matches_optimal == -1:
        u_b, u_l, c, q, lambda_1, lambda_2, lambda_3, preference_borrowers, preference_lenders = simulate_lending()
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
                                                                                      lambda_2,  preference_borrowers,
                                                                                      preference_lenders, NUM_SIMS_PER_STEP, T, rewards_from_borrowers, VARIANCE)
    # for b_idx in borrower_matches:
    #     sum_l = 0
    #     print("Borrower ", b_idx, " with request ", c[b_idx])
    #     for l_idx in borrower_matches[b_idx]:
    #         print("Lender", l_idx, " invested ", q[l_idx])
    #         sum_l += q[l_idx]
    #     print("Total amount raised by borrower ", b_idx, " is: ", sum_l)

    plot_lines(regret_lenders_basic, save_dir="../figures/figure_3_20/20b_60l/basic")

    print("-----------------------------------GS BLEMET -----------------------------------")
    regret_lenders_blemet, borrower_matches, lender_matches, objVal = gs_bandit_BLEMET(u_b, u_l, c, q, lambda_1,
                                                                                      lambda_2,  lambda_3, NUM_SIMS_PER_STEP, T, rewards_from_borrowers, VARIANCE)
    # for b_idx in borrower_matches:
    #     sum_l = 0
    #     print("Borrower ", b_idx, " with request ", c[b_idx])
    #     for l_idx in borrower_matches[b_idx]:
    #         print("Lender", l_idx, " invested ", q[l_idx])
    #         sum_l += q[l_idx]
    #     print("Total amount raised by borrower ", b_idx, " is: ", sum_l)

    plot_lines_blemet(regret_lenders_blemet, save_dir="../figures/figure_3_20/20b_60l/phases/")

    regret_lenders_blemet_fairness, borrower_matches, lender_matches, objVal = gs_bandit_BLEMET(u_b, u_l, c, q, lambda_1,
                                                                                       lambda_2, lambda_3,
                                                                                       NUM_SIMS_PER_STEP, T,
                                                                                       rewards_from_borrowers, with_fairness=True, VARIANCE=VARIANCE)
    plot_lines_aggregate(regret_lenders_basic, regret_lenders_blemet, "GS-UCB", "GS-BLEMET",
                         save_dir="../figures/figure_3_20/20b_60l/agg/")
    plot_lines_aggregate(regret_lenders_blemet_fairness, regret_lenders_blemet, "GS-FAIR", "GS-BLEMET", save_dir="../figures/figure_3_20/20b_60l/fairness/")

