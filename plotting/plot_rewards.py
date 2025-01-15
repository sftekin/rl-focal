import os
import sys

sys.path.append("..")

import glob
import numpy as np
import matplotlib.pyplot as plt
from config import RESULTS_DIR
from run import load_arr


# plt.style.use('seaborn-v0_8')
# Set global font size for all elements
plt.rcParams.update({
    'font.size': 14,              # General font size
    'axes.titlesize': 16,         # Title font size
    'axes.labelsize': 16,         # Axis label font size
    'xtick.labelsize': 14,        # X-axis tick font size
    'ytick.labelsize': 14,        # Y-axis tick font size
    'legend.fontsize': 14         # Legend font size
})


def fix_arr(in_list):
    new_arr_list = []
    max_size = max([len(arr) for arr in in_list])
    for input_arr in in_list:
        if len(input_arr) > 0:
            pad_size = max_size - len(input_arr)
            if pad_size > 0:
                pad_arr = np.empty(pad_size)
                pad_arr[:] = np.nan
                new_arr = np.concatenate([input_arr, pad_arr]) 
            else:
                new_arr = input_arr
            new_arr_list.append(new_arr)
    return np.stack(new_arr_list)


def main():
    task_names = ["mmlu_hf", "gsm8k", "bbh", "gpqa", "musr"]
    colors = [f"C{i}" for i in range(len(task_names))]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for i, task_name in enumerate(task_names):
        checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints", task_name)
        select_agent_list = []
        ens_agent_list = []
        for file_dir in glob.glob(f"{checkpoint_dir}/*"):
            ens_agent_path = os.path.join(file_dir, "train_ens_agent_rewards.npy")
            select_agent_path = os.path.join(file_dir, "train_select_agent_rewards.npy")
            if os.path.exists(select_agent_path):
                select_agent_list.append(load_arr(select_agent_path))
                ens_agent_list.append(load_arr(ens_agent_path))
                print(select_agent_path, len(load_arr(ens_agent_path)))

        select_agent_arr = fix_arr(select_agent_list)
        ens_agent_arr = fix_arr(ens_agent_list)
    # print(ens_agent_arr)

        
        mu = np.nanmean(select_agent_arr, axis=0)
        sigma = np.nanstd(select_agent_arr, axis=0)
        x_axis = np.arange(len(mu))
        ax[0].plot(x_axis, mu, label=task_name.split("_")[0].upper(), alpha=0.8, color=colors[i], zorder=2)
        ax[0].fill_between(x_axis, mu + sigma, mu - sigma, alpha=0.3, color=colors[i], zorder=2)
        ax[0].set_ylim(0, 85)
        # ax[0].legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax[0].set_xlabel("Episode Number")
        ax[0].set_ylabel("Accuracy (%)")
        ax[0].set_title(f"Select Agent Accuracy per Episode")
        ax[0].yaxis.grid(zorder=0)
        # plt.show()

        mu = np.nanmean(ens_agent_arr, axis=0)
        sigma = np.nanstd(ens_agent_arr, axis=0)
        x_axis = np.arange(len(mu))
        ax[1].plot(x_axis, mu, label=task_name.split("_")[0].upper(), alpha=0.8, color=colors[i], zorder=2)
        ax[1].fill_between(x_axis, mu + sigma, mu - sigma, alpha=0.3, color=colors[i], zorder=2)
        ax[1].set_ylim(0, 85)
        ax[1].legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax[1].set_xlabel("Episode Number")
        ax[1].set_ylabel("Accuracy (%)")
        ax[1].set_title(f"Ensemble Agent Accuracy per Episode")
        ax[1].yaxis.grid(zorder=0)
    # plt.show()

    plt.savefig(f"../results/figures/reward_plot_side.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()