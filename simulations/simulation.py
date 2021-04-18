import random
from models.decision_making.seq_decision_making import *


def simulate_lending():
    u_b = {}
    u_l = {}

    n_b, n_l = 20, 60

    preference_borrowers = []
    preference_lenders = []

    sim_values = {}
    for b_idx in range(1, n_b + 1):
        sim_values[b_idx] = {}
        for l_idx in range(1, n_l + 1):
            sim_values[b_idx][l_idx] = random.uniform(0, 0.5)

    borrower_rates = {}
    for b_idx in range(1, n_b + 1):
        borrower_rates[b_idx] = random.uniform(0.1, 0.5)   # consider fixed for now, later random.uniform(0, 1)

    # This part is not used for now - risk_preference
    risk_preference = {}
    categories = list(range(1, 6))
    lender_risk_categories = {}
    for l_idx in range(1, n_l + 1):
        lender_risk_categories[l_idx] = {}
        for c_idx in categories:
            lender_risk_categories[l_idx][c_idx] = random.uniform(0, 0.5)
    borrower_categories = {}
    for b_idx in range(1, n_b + 1):
        borrower_categories[b_idx] = random.sample(categories, 1)[0]  # consider every borrower has 1 category for now
        risk_preference[b_idx] = {}
        for l_idx in range(1, n_l + 1):
            risk_preference[b_idx][l_idx] = lender_risk_categories[l_idx][borrower_categories[b_idx]]

    # c - borrower amount, q - lender amount
    c = {}
    q = {}

    while True:
        sum_c = 0
        sum_q = 0
        for b_idx in range(1, n_b + 1):
            c[b_idx] = random.sample(range(10, 50), 1)[0]
            sum_c += c[b_idx]

        for l_idx in range(1, n_l + 1):
            q[l_idx] = random.sample(range(1, 30), 1)[0]
            sum_q += q[l_idx]

        print(sum_q, sum_c)
        if sum_q > sum_c:
            break

    # utilities
    for b_idx in range(1, n_b + 1):
        u_b[b_idx] = {}
        for l_idx in range(1, n_l + 1):
            u_b[b_idx][l_idx] = borrower_utility(l_idx, b_idx, sim_values, risk_preference)
        preference_borrowers = sorted(range(1, len(u_b[b_idx]) + 1), key=lambda k: u_b[b_idx][k])

    for l_idx in range(1, n_l + 1):
        u_l[l_idx] = {}
        for b_idx in range(1, n_b + 1):
            u_l[l_idx][b_idx] = lender_utility(l_idx, b_idx, sim_values, q, borrower_rates)
        preference_lenders = sorted(range(1, len(u_l[l_idx]) + 1), key=lambda k: u_l[l_idx][k])


    print("Configuration:")
    print("Borrower preferences: ", u_b)
    print("Borrower requests: ", c)
    print("Lender preferences: ", u_l)
    print("Lender budgets: ", q)

    return u_b, u_l, c, q, preference_borrowers, preference_lenders
