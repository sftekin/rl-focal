import os
import numpy as np
import matplotlib.pyplot as plt
from config import CUR_DIR

def plot_actions(action_count, count, m_names, ep_count, ax):
    plots_dir = os.path.join(CUR_DIR, "results", "figures")
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
