import os
import numpy as np
import matplotlib.pyplot as plt
from config import CUR_DIR
import itertools
# from ..env.ens_metrics import calc_div_acc

plots_dir = os.path.join(CUR_DIR, "results", "figures")


def plot_actions(action_count, count, m_names, ep_count, ax):
    # fig, ax = plt.subplots()
    x_axis = np.arange(len(m_names))
    y_axis = action_count / count
    print(y_axis)
    ax.bar(x_axis, y_axis)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(m_names, rotation=90)
    save_path = os.path.join(plots_dir, f"action_dist_{ep_count}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    ax.cla()



def plot_rewards(rewards, label):
    fig, ax = plt.subplots()
    x_axis = np.arange(len(rewards))
    ax.plot(x_axis, rewards, "--", marker=">", lw=2, label=label)
    save_path = os.path.join(plots_dir, f"rewards_{label}.png")
    ax.set_xlabel("Episode Number")
    ax.set_ylabel("Cummulative Reward")
    ax.grid()
    ax.legend()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    ax.cla()


def plot_contour(hist_data, task_name, num_models):
    scores = []
    ens_sizes = np.arange(2, num_models + 1)
    for j, ens_size in enumerate(ens_sizes):
        print(ens_size)
        combinations = list(itertools.combinations(range(num_models), ens_size))
        for comb in combinations:
            comb_idx = np.zeros(num_models, dtype=int)
            comb_idx[list(comb)] = 1
            scores.append(calc_div_acc(comb_idx, hist_data))
    scores = np.array(scores)

    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    plt.style.use('default')

    # Set global font size for all elements
    plt.rcParams.update({
        'font.size': 14,              # General font size
        'axes.titlesize': 16,         # Title font size
        'axes.labelsize': 16,         # Axis label font size
        'xtick.labelsize': 12,        # X-axis tick font size
        'ytick.labelsize': 12,        # Y-axis tick font size
        'legend.fontsize': 12         # Legend font size
    })

    X = scores[:, 0]
    Y = scores[:, 2]
    Z = scores[:, 1]
    xi = np.linspace(X.min(),X.max(),100)
    yi = np.linspace(Y.min(),Y.max(),100)
    zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='cubic')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    xig, yig = np.meshgrid(xi, yi)
    surf = ax.plot_surface(xig, yig, zi, cmap=cm.coolwarm,
                    linewidth=1, antialiased=False, zorder=1)
    ax.set_xlabel(r"Focal Diversity $\lambda$")
    ax.set_ylabel(r"Fleiss Kappa $\kappa$")
    ax.set_zlabel("Accuracy (%)")

    # Customize the z axis.
    # ax.set_zlim(0.30, Z.max() + 0.05)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')
    ax.view_init(elev=20., azim=-120)
    ax.set_title(f"{task_name}")

    plt.savefig(f"contour_{task_name}.png", bbox_inches="tight", dpi=250)

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

