from simulations.simulation import *
from utilities.plots import *
import multiprocessing
import os
import dill

VARIANCE = 0.2
NUM_SIMS_PER_STEP = 1
T = 50 # horizon, n
SAVE_DIR = "../figures/figure_02_06_0p5_var1/100b_20l"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR + "/phases/", exist_ok=True)
os.makedirs(SAVE_DIR + "/phases_fair/", exist_ok=True)
os.makedirs(SAVE_DIR + "/agg/", exist_ok = True)
os.makedirs(SAVE_DIR + "/basic/", exist_ok = True)
os.makedirs(SAVE_DIR + "/comp/", exist_ok = True)
os.makedirs(SAVE_DIR + "/comp_bl/", exist_ok = True)
os.makedirs(SAVE_DIR + "/comp_max/", exist_ok = True)
os.makedirs(SAVE_DIR + "/hmap_basic/", exist_ok = True)
os.makedirs(SAVE_DIR + "/hmap_basic_mp/", exist_ok = True)
os.makedirs(SAVE_DIR + "/comp_borrowers/", exist_ok = True)


if __name__ == "__main__":
    fns = []
    borrower_matches_optimal = -1
    print("-----------------------------------Optimal matching-----------------------------------")
    while borrower_matches_optimal == -1:
        alpha_t, u_b, u_l, c, q, preference_borrowers, preference_lenders = simulate_lending()
        rewards_from_borrowers = get_rewards_list_start(len(u_l), len(u_b), u_b, NUM_SIMS_PER_STEP, T, VARIANCE)

        lambda_1 = 0.5
        lambda_2 = 0.5
        borrower_matches_optimal, lender_matches_optimal, objVal = gs_bandit_method_baseline(alpha_t, u_b, u_l, c, q, lambda_1,
                                                                                             lambda_2)  # baseline

    print("-----------------------------------GS Basic-----------------------------------")
    regret_lenders_basic, arms_ts, arms_ts_mostpref, borrower_matches, lender_matches, objVal = gs_bandit_method_basic(
                                                                                                alpha_t, u_b, u_l, c, q, lambda_1,
                                                                                                lambda_2,  preference_borrowers,
                                                                                                preference_lenders, NUM_SIMS_PER_STEP, T,
                                                                                                rewards_from_borrowers, VARIANCE,
                                                                                                objective_type="lenders")
    plot_lines(regret_lenders_basic, u_l, save_dir=SAVE_DIR + "/basic/")
