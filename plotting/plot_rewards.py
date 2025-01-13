import os
import sys

sys.path.append("..")

import glob
import numpy as np
import matplotlib.pyplot as plt
from config import RESULTS_DIR
from run import load_arr


plt.style.use('seaborn-v0_8')
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
    task_name = "mmlu_hf"
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints", task_name)
    select_agent_list = []
    ens_agent_list = []
    for file_dir in glob.glob(f"{checkpoint_dir}/*"):
        ens_agent_path = os.path.join(file_dir, "train_ens_agent_rewards.npy")
        select_agent_path = os.path.join(file_dir, "train_select_agent_rewards.npy")
        if os.path.exists(select_agent_path):
            select_agent_list.append(load_arr(select_agent_path))
            ens_agent_list.append(load_arr(ens_agent_path))

    select_agent_arr = fix_arr(select_agent_list)
    ens_agent_arr = fix_arr(ens_agent_list)
    print(ens_agent_arr)

    fig, ax = plt.subplots()
    line_names = ["Select Agent", "Ensemble Agent"]
    colors  = ["tab:red", "tab:blue"]
    for i, arr in enumerate([select_agent_arr, ens_agent_arr]):
        mu = np.nanmean(arr, axis=0)
        sigma = np.nanstd(arr, axis=0)
        x_axis = np.arange(len(mu))
        ax.plot(x_axis, mu, color=colors[i], label=line_names[i], alpha=0.8)
        ax.fill_between(x_axis, mu + sigma, mu - sigma, color=colors[i], alpha=0.3)
    ax.set_ylim(15, 85)
    ax.legend(fontsize=16)
    ax.set_xlabel("Episode Number")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Agents Accuracy per Episode in {task_name.upper()}")
    plt.savefig(f"../results/figures/reward_plot_{task_name}.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()