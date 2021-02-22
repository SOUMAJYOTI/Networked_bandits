from simulations.simulation import *
from utilities.plots import *

if __name__ == "__main__":
    borrower_matches_optimal = -1

    while borrower_matches_optimal == -1:
        u_b, u_l, c, q, lambda_1, lambda_2, preference_borrowers, preference_lenders = simulate_lending()
        borrower_matches_optimal, lender_matches_optimal, objVal = gs_bandit_method_baseline(u_b, u_l, c, q, lambda_1,
                                                                                             lambda_2)  # baseline

    print("-----------------------------------Optimal matching-----------------------------------")
    for b_idx in borrower_matches_optimal:
        sum_l = 0
        print("Borrower ", b_idx, " with request ", c[b_idx])
        for l_idx in borrower_matches_optimal[b_idx]:
            print("Lender", l_idx, " invested ", q[l_idx])
            sum_l += q[l_idx]
        print("Total amount raised by borrower ", b_idx, " is: ", sum_l)

    # print("-----------------------------------GS Basic-----------------------------------")
    # regret_lenders, borrower_matches, lender_matches, objVal = gs_bandit_method_basic(u_b, u_l, c, q, lambda_1,
    #                                                                                   lambda_2, preference_borrowers,
    #                                                                                   preference_lenders)
    # for b_idx in borrower_matches:
    #     sum_l = 0
    #     print("Borrower ", b_idx, " with request ", c[b_idx])
    #     for l_idx in borrower_matches[b_idx]:
    #         print("Lender", l_idx, " invested ", q[l_idx])
    #         sum_l += q[l_idx]
    #     print("Total amount raised by borrower ", b_idx, " is: ", sum_l)
    #
    # plot_lines(regret_lenders, save_dir="../figures/figure_2_21")

    print("-----------------------------------GS Phases-----------------------------------")
    regret_lenders, borrower_matches, lender_matches, objVal = gs_bandit_phases(u_b, u_l, c, q, lambda_1,
                                                                                      lambda_2, preference_borrowers,
                                                                                      preference_lenders)
    # for b_idx in borrower_matches:
    #     sum_l = 0
    #     print("Borrower ", b_idx, " with request ", c[b_idx])
    #     for l_idx in borrower_matches[b_idx]:
    #         print("Lender", l_idx, " invested ", q[l_idx])
    #         sum_l += q[l_idx]
    #     print("Total amount raised by borrower ", b_idx, " is: ", sum_l)

    plot_lines(regret_lenders, save_dir="../figures/figure_2_21/phases/")
