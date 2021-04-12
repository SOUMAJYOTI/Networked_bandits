import matplotlib.pyplot as plt
import numpy as np


def plot_lines(regret, u_l, save_dir="."):
    for l_idx in regret:
        time_points = []
        time_points_ub = []
        time_points_lb = []
        max_ub = -1
        min_lb = 10

        for t_idx in regret[l_idx]:
            # print(regret[l_idx][t_idx])
            # print("ub", regret[l_idx][t_idx])

            time_points.append(np.mean(regret[l_idx][t_idx]))
            diff = np.mean(regret[l_idx][t_idx]) - np.mean(regret[l_idx][1])
            time_points_ub.append(np.mean(regret[l_idx][t_idx]) + 0.1*diff)
            time_points_lb.append(np.mean(regret[l_idx][t_idx]) - 0.1*diff)

        x = range(1, len(time_points)+1)
        plt.plot(x, time_points, 'k-', linewidth=3)
        plt.fill_between(x, time_points_lb, time_points_ub, color='orange')
        # print(max_ub, min_lb)
        # plt.ylim([max_ub + abs(max_ub) , min_lb - abs(min_lb) ])
        plt.xlabel("Time steps", size=20)
        plt.ylabel("Regret", size=20)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.title("Lender " + str(l_idx), size=20)
        plt.subplots_adjust(left=0.18, bottom=0.17, top=0.85, right=0.9)
        plt.savefig(save_dir + "/" + "lender_" + str(l_idx) + ".png")
        plt.close()


def plot_lines_blemet(regret, u_l, save_dir="."):
    for l_idx in regret:
        time_points = []
        time_points_ub = []
        time_points_lb = []
        max_ub = -1
        min_lb = 10

        for t_idx in regret[l_idx]:
            # print(regret[l_idx][t_idx])
            # print("ub", regret[l_idx][t_idx])

            # print(regret[l_idx][t_idx])
            time_points.append(np.mean(regret[l_idx][t_idx]))
            diff = np.mean(regret[l_idx][t_idx]) - np.mean(regret[l_idx][1])
            time_points_ub.append(np.mean(regret[l_idx][t_idx]) + 0.1 * diff)
            time_points_lb.append(np.mean(regret[l_idx][t_idx]) - 0.1 * diff)

        x = range(1, len(time_points)+1)
        plt.plot(x, time_points, 'k-', linewidth=3)
        plt.fill_between(x, time_points_lb, time_points_ub, color='orange')
        # print(max_ub, min_lb)
        # plt.ylim([max_ub + abs(max_ub) , min_lb - abs(min_lb) ])
        plt.xlabel("Time steps", size=20)
        plt.ylabel("Regret", size=20)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.title("Lender " + str(l_idx), size=20)
        plt.subplots_adjust(left=0.18, bottom=0.17, top=0.85, right=0.9)
        plt.savefig(save_dir + "/" + "lender_" + str(l_idx)  + ".png")
        plt.close()


def plot_lines_aggregate(regret_basic, regret_blemet, u_l,  label1, label2, save_dir="."):
    for l_idx in regret_basic:
        time_points_basic = []
        time_points_ub_basic = []
        time_points_lb_basic = []

        time_points_blemet = []
        time_points_ub_blemet = []
        time_points_lb_blemet = []

        max_ub = -1
        min_lb = 10

        for t_idx in regret_basic[l_idx]:

            time_points_basic.append(np.mean(regret_basic[l_idx][t_idx]))
            diff = np.mean(regret_basic[l_idx][t_idx]) - np.mean(regret_basic[l_idx][1])
            time_points_ub_basic.append(np.mean(np.mean(regret_basic[l_idx][t_idx]) + 0.1 * diff))
            time_points_lb_basic.append(np.mean(np.mean(regret_basic[l_idx][t_idx]) - 0.1 * diff))

            time_points_blemet.append(np.mean(regret_blemet[l_idx][t_idx]))
            diff = np.mean(regret_blemet[l_idx][t_idx]) - np.mean(regret_blemet[l_idx][1])
            time_points_ub_blemet.append(np.mean(np.mean(regret_blemet[l_idx][t_idx]) + 0.1 * diff))
            time_points_lb_blemet.append(np.mean(np.mean(regret_blemet[l_idx][t_idx]) - 0.1 * diff))

            # if (np.mean(regret_basic[l_idx][t_idx]) + np.std(regret_basic[l_idx][t_idx])) > max_ub:
            #     max_ub = np.mean(regret_basic[l_idx][t_idx]) + np.power(np.std(regret_basic[l_idx][t_idx]), 2)
            #
            # if (np.mean(regret_basic[l_idx][t_idx]) - np.std(regret_basic[l_idx][t_idx])) < min_lb:
            #     min_lb = np.mean(regret_basic[l_idx][t_idx]) - np.power(np.std(regret_basic[l_idx][t_idx]), 2)

        x = range(len(time_points_basic))
        plt.plot(x, time_points_basic, 'k-', lw=3)
        plt.fill_between(x, time_points_lb_basic, time_points_ub_basic, color='green', label=label1)
        plt.plot(x, time_points_blemet, 'k-', lw=3)
        plt.fill_between(x, time_points_lb_blemet, time_points_ub_blemet, color='orange', label=label2)
        # plt.ylim([max_ub +  2*max_ub, min_lb - 2*min_lb])
        plt.xlabel("Time steps", size=20)
        plt.ylabel("Regret", size=20)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.legend(loc='upper right')
        plt.title("Lender " + str(l_idx), size=20)
        plt.subplots_adjust(left=0.18, bottom=0.17, top=0.85, right=0.9)
        plt.savefig(save_dir + "/" + "lender_" + str(l_idx) + ".png")
        plt.close()


def plot_lines_sum_regrets(regret_basic, regret_blemet,  label1, label2, save_dir="."):
    time_points_basic = []
    time_points_ub_basic = []
    time_points_lb_basic = []

    time_points_blemet = []
    time_points_ub_blemet = []
    time_points_lb_blemet = []

    for t_idx in regret_basic:
        list_r = list(regret_basic[t_idx].values())
        time_points_basic.append(np.mean(list_r))
        diff = np.mean(list_r) - np.mean(list(regret_basic[1].values()))
        time_points_ub_basic.append(np.mean(np.mean(list_r) + 0.1 * diff))
        time_points_lb_basic.append(np.mean(np.mean(list_r) - 0.1 * diff))

        list_r = list(regret_blemet[t_idx].values())
        time_points_blemet.append(np.mean(list_r))
        diff = np.mean(list_r) - np.mean(list(regret_blemet[1].values()))
        time_points_ub_blemet.append(np.mean(np.mean(list_r) + 0.1 * diff))
        time_points_lb_blemet.append(np.mean(np.mean(list_r) - 0.1 * diff))

    x = range(len(time_points_basic))
    plt.plot(x, time_points_basic, 'k-', lw=3)
    plt.fill_between(x, time_points_lb_basic, time_points_ub_basic, color='green', label=label1)
    plt.plot(x, time_points_blemet, 'k-', lw=3)
    plt.fill_between(x, time_points_lb_blemet, time_points_ub_blemet, color='orange', label=label2)
    # plt.ylim([max_ub +  2*max_ub, min_lb - 2*min_lb])
    plt.xlabel("Time steps", size=20)
    plt.ylabel("Regret", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.legend(loc='upper right')
    plt.title("Sum of lender regrets", size=20)
    plt.subplots_adjust(left=0.18, bottom=0.17, top=0.85, right=0.9)
    plt.savefig(save_dir + "/" + "lender_regret.png")
    plt.close()
