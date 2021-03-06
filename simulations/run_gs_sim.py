from simulations.simulation import *
from utilities.plots import *
import os

VARIANCE = 1.0
NUM_SIMS_PER_STEP = 5
T = 10000 # horizon, n
SAVE_DIR = "../figures/figure_4_18_1p5_var1/20b_60l"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR + "/phases/", exist_ok=True)
os.makedirs(SAVE_DIR + "/phases_fair/", exist_ok=True)
os.makedirs(SAVE_DIR + "/agg/", exist_ok = True)
os.makedirs(SAVE_DIR + "/basic/", exist_ok = True)
os.makedirs(SAVE_DIR + "/comp/", exist_ok = True)
os.makedirs(SAVE_DIR + "/comp_max/", exist_ok = True)


if __name__ == "__main__":
    borrower_matches_optimal = -1
    print("-----------------------------------Optimal matching-----------------------------------")
    while borrower_matches_optimal == -1:
        u_b, u_l, c, q, preference_borrowers, preference_lenders = simulate_lending()
        rewards_from_borrowers = get_rewards_list_start(len(u_l), len(u_b), u_b, NUM_SIMS_PER_STEP, T, VARIANCE)
        lambda_1 = 0.5
        lambda_2 = 0.5
        borrower_matches_optimal, lender_matches_optimal, objVal = gs_bandit_method_baseline(u_b, u_l, c, q, lambda_1,
                                                                                             lambda_2)  # baseline

    print("-----------------------------------GS Basic-----------------------------------")
    regret_lenders_basic, borrower_matches, lender_matches, objVal = gs_bandit_method_basic(u_b, u_l, c, q, lambda_1,
                                                                                      lambda_2,  preference_borrowers,
                                                                                      preference_lenders, NUM_SIMS_PER_STEP, T, rewards_from_borrowers, VARIANCE)

    plot_lines(regret_lenders_basic,q, save_dir=SAVE_DIR + "/basic")


    print("-----------------------------------GS BLEMET -----------------------------------")
    lambda_1 = 0.5
    lambda_2 = 0.5
    lambda_3 = 1 - (lambda_1 + lambda_2)

    regret_lenders_blemet, borrower_matches, lender_matches, objVal = gs_bandit_BLEMET(u_b, u_l, c, q, lambda_1,
                                                                                      lambda_2,  lambda_3, NUM_SIMS_PER_STEP, T, rewards_from_borrowers,
                                                                                       with_fairness=False, VARIANCE=VARIANCE)

    plot_lines_blemet(regret_lenders_blemet, q, save_dir=SAVE_DIR + "/phases/")

    print("-----------------------------------GS BLEMET FAIR -----------------------------------")
    lambda_1 = 0.5
    lambda_2 = 0.25
    lambda_3 = 1 - (lambda_1 + lambda_2)
    regret_lenders_blemet_fair, borrower_matches, lender_matches, objVal = gs_bandit_BLEMET(u_b, u_l, c, q, lambda_1,
                                                                                       lambda_2, lambda_3,
                                                                                       NUM_SIMS_PER_STEP, T,
                                                                                       rewards_from_borrowers, with_fairness=True, VARIANCE=VARIANCE)

    plot_lines_blemet(regret_lenders_blemet_fair, q, save_dir=SAVE_DIR + "/phases_fair/")

    plot_lines_aggregate(regret_lenders_basic, regret_lenders_blemet, regret_lenders_blemet_fair, "GS-UCB", "GS-BLEMET", "GS-BLEMET-FAIR",
                         save_dir=SAVE_DIR + "/agg/")


    regret_basic_sum = sum_regrets(regret_lenders_basic)
    regret_blemet_sum = sum_regrets(regret_lenders_blemet)
    regret_blemet_fair_sum = sum_regrets(regret_lenders_blemet_fair)

    regret_basic_max = max_regrets(regret_lenders_basic)
    regret_blemet_max = max_regrets(regret_lenders_blemet)
    regret_blemet_fair_max = max_regrets(regret_lenders_blemet_fair)

    plot_lines_sum_regrets(regret_basic_sum, regret_blemet_sum, regret_blemet_fair_sum, "GS-UCB", "GS-BLEMET", "GS-BLEMET-FAIR",
                           save_dir=SAVE_DIR + "/comp/")

    plot_lines_sum_regrets(regret_basic_max, regret_blemet_max, regret_blemet_fair_max, "GS-UCB", "GS-BLEMET", "Gs-BLEMET-FAIR",
                           save_dir=SAVE_DIR + "/comp_max/")