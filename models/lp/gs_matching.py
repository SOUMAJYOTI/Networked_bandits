from gurobipy import *


def model_gs_matching(u_b, u_l, c, q, obj_util, lambda_1, lambda_2, LogToConsole=True, TimeLimit=60):
    model = Model()
    model.params.LogToConsole = LogToConsole
    model.params.TimeLimit = TimeLimit  # seconds
    x = {}
    w = {}
    for b_idx in c:
        x[b_idx] = {}
        w[b_idx] = {}
        for l_idx in q:
            x_name = "x_{}_{}".format(b_idx, l_idx)
            x[b_idx][l_idx] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=x_name)
            w_name = "w_{}_{}".format(b_idx, l_idx)
            w[b_idx][l_idx] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=w_name)

    for l_idx in q:
        model.addConstr(quicksum(x[b_idx][l_idx] for b_idx in c) <= 1)

    for b_idx in c:
        model.addConstr(quicksum((q[l_idx] * x[b_idx][l_idx]) for l_idx in q) >= c[b_idx])

    for b_idx in c:
        for l_idx in q:
            constr_obj_1 = c[b_idx] * x[b_idx][l_idx]
            constr_obj_2 = 0
            constr_obj_3 = 0

            for b_idx_2 in c:
                if b_idx != b_idx_2:
                    if u_l[l_idx][b_idx] < u_l[l_idx][b_idx_2]:
                        constr_obj_2 += (x[b_idx][l_idx])
            constr_obj_2 *= c[b_idx]

            for l_idx_2 in q:
                if l_idx != l_idx_2:
                    if u_b[b_idx][l_idx] < u_b[b_idx][l_idx_2]:
                        constr_obj_3 += (q[l_idx] * x[b_idx][l_idx])

            model.addConstr((constr_obj_1 + constr_obj_2 + constr_obj_3) >= (c[b_idx] * (1 - w[b_idx][l_idx])))

    model.setObjective(lambda_1 * quicksum(obj_util[b_idx][l_idx] * x[b_idx][l_idx] for l_idx in q for b_idx in c) - \
                       lambda_2 * quicksum(w[b_idx][l_idx] for l_idx in q for b_idx in c), GRB.MAXIMIZE)
    model.optimize()
    if model.status != 2:
        print("Optimal Solution not found !!!")
        return -1, -1, -1

    borrower_matches = {}
    lender_matches = {}
    for b_idx in c:
        borrower_matches[b_idx] = []
        #print("Borrower {} matched to lenders: ".format(b_idx))
        for l_idx in q:
            if x[b_idx][l_idx].X == 1:
                #                 print(l_idx)
                borrower_matches[b_idx].append(l_idx)
                if l_idx not in lender_matches:
                    lender_matches[l_idx] = -1
                lender_matches[l_idx] = b_idx

    return borrower_matches, lender_matches, model.objVal
