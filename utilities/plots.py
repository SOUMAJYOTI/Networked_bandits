import matplotlib.pyplot as plt
import numpy as np


def plot_lines(regret, save_dir="."):
    for l_idx in regret:
        time_points = []
        time_points_ub = []
        time_points_lb = []
        max_ub = -1
        min_lb = 10

        for t_idx in regret[l_idx]:
            time_points.append(np.mean(regret[l_idx][t_idx]))
            time_points_ub.append(np.mean(regret[l_idx][t_idx]) + np.power(np.std(regret[l_idx][t_idx]), 2))
            time_points_lb.append(np.mean(regret[l_idx][t_idx]) - np.power(np.std(regret[l_idx][t_idx]), 2))

            if (np.mean(regret[l_idx][t_idx]) + np.std(regret[l_idx][t_idx])) > max_ub:
                max_ub = np.mean(regret[l_idx][t_idx]) + np.power(np.std(regret[l_idx][t_idx]), 2)

            if (np.mean(regret[l_idx][t_idx]) - np.std(regret[l_idx][t_idx])) < min_lb:
                min_lb = np.mean(regret[l_idx][t_idx]) - np.power(np.std(regret[l_idx][t_idx]), 2)

        x = range(len(time_points))
        plt.plot(x, time_points, 'k-')
        plt.fill_between(x, time_points_lb, time_points_ub)
        plt.ylim([max_ub +  max_ub, min_lb - min_lb])
        plt.xlabel("Time steps", size=20)
        plt.ylabel("Regret", size=20)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.title("Lender " + str(l_idx), size=20)
        plt.savefig(save_dir + "/" + "lender_img" + str(l_idx) + ".png")
        plt.close()
