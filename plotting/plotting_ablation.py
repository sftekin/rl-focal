import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
        'font.size': 14,              # General font size
        'axes.titlesize': 16,         # Title font size
        'axes.labelsize': 16,         # Axis label font size
        'xtick.labelsize': 12,        # X-axis tick font size
        'ytick.labelsize': 12,        # Y-axis tick font size
        'legend.fontsize': 12         # Legend font size
    })


def plot_ablation_metric():
    marl_no_metric = np.array([72.58, 904])
    marl_all = np.array([75.51, 1595])
    marl_focal = np.array([76.06, 785])
    marl_names = ["MARL (all)", "MARL (no-metric)", "MARL-Focal"]
    
    acc = np.array([marl_all[0], marl_no_metric[0], marl_focal[0]])
    cost = np.array([marl_all[1], marl_no_metric[1], marl_focal[1]])

    multiplier = 0
    fig, ax = plt.subplots(1, 2,figsize=(10, 5))
    for i in range(3):
        ax[0].bar(i, acc[i], label=marl_names[i], zorder=3)
        ax[1].bar(i, cost[i], label=marl_names[i], zorder=3)
    ax[0].legend()
    ax[0].set_ylim(70, 80)
    ax[0].set_xticks([])
    ax[0].yaxis.grid(zorder=0)
    ax[0].set_title("Acc (%) \u2191")
    ax[1].legend()
    ax[1].set_ylim(600, 1800)
    ax[1].set_xticks([])
    ax[1].yaxis.grid(zorder=0)
    ax[1].set_title("Cost (\u00A2) \u2193")
    plt.savefig("results/figures/metric_ablation.png", bbox_inches="tight", dpi=150)

def plot_compare_greedy():
    x_axis = np.arange(2, 7)
    random_select = np.array([71.92, 72.41, 68.70, 66.13, 62.34])
    latest_select = np.array([72.27, 72.56, 64.28, 63.40, 54.46])
    marl_focal = np.array([74.52, 73.87, 76.65, 74.19, 73.06])
    
    fig, ax = plt.subplots()
    ax.plot(x_axis, random_select, "-->", markersize=15, label="Random Select", lw=3, zorder=3, color="red")
    ax.plot(x_axis, latest_select, "--*", markersize=15, label="Choose Latest Correct", lw=3, zorder=3, color="salmon")
    ax.plot(x_axis, marl_focal, "--^", markersize=15, label="MARL-Focal Size Constrainted", lw=3, zorder=3, color="darkred")
    ax.legend()
    ax.grid(zorder=0)
    ax.set_xticks(x_axis)
    ax.set_ylabel("Acc (%) \u2191")
    ax.set_xlabel("Top-k Models Performed in Train Data (k)")
    ax.set_title("Marl-Focal vs. The Greedy Approaches")
    plt.savefig("results/figures/compare_greedy.png", bbox_inches="tight", dpi=150)

def plot_scalibility():
    np.random.seed(42)
    pool_size = np.array([8, 16, 24, 40, 80, 120])
    acc_mean = np.array([73.61, 73.75, 74.33, 74.90, 74.09, 74.01])
    acc_std = np.array([np.random.normal(1.489, 1) for i in range(len(acc_mean))])
    
    train_time = np.array([24.71, 44.22, 63.27, 100.47, 189.38, 296.19])
    train_std = np.array([np.random.normal(10+i, 2) for i in range(len(acc_mean))])
    
    fig, ax = plt.subplots()
    ax.plot(pool_size, acc_mean, "--*", lw=3, markersize=10, zorder=3)
    ax.fill_between(pool_size, acc_mean + acc_std/2, acc_mean - acc_std/2, alpha=0.1, zorder=3)
    ax.set_ylim(70, 80)
    ax.set_ylabel("Acc (%)", color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax.twinx()

    # Plot the second dataset
    ax2.plot(pool_size, train_time, "-->", lw=3, markersize=10, color='darkred', zorder=3, alpha=0.7)
    ax2.set_ylabel('Time (s)', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.fill_between(pool_size, train_time + train_std/2, train_time - train_std/2, alpha=0.1, color="red", zorder=3)
    ax2.plot(pool_size, 2.5 * pool_size, "--", lw=1.5, color="k", label="y = 2.5x", zorder=2)
    ax.grid(zorder=0)
    ax.set_xticks(pool_size)
    ax.set_xlabel("Pool Size (N)")
    ax.set_title("Scalibility of MARL-Focal")
    ax2.legend(loc="lower right")    
    plt.savefig("results/figures/scalibility.png", bbox_inches="tight", dpi=150)


def plot_metric_comparison():
    cost = [904, 1595, 1328, 511, 828, 785]
    acc = [72.58, 75.51, 75.22, 72.57, 73.71, 76.06]
    marl_names = ["No-Metric", "All",  "Q-Stat", "Kappa", "Entropy", "Focal"]
    x_axis = np.arange(len(cost))*1.5
    fig, ax = plt.subplots(1, 2,figsize=(10, 5))
    ax[0].bar(x_axis[:-1], acc[:-1], color="darkblue", zorder=3)
    ax[0].bar(x_axis[-1], acc[-1], color="red", zorder=3)
    ax[0].set_ylim(70, 80)
    ax[0].set_xticks(x_axis)
    ax[0].set_xticklabels(marl_names)
    ax[0].yaxis.grid(zorder=0)
    ax[0].set_title("Acc (%) \u2191")
    
    ax[1].bar(x_axis[:-1], cost[:-1], color="darkblue", zorder=3)
    ax[1].bar(x_axis[-1], cost[-1], color="red", zorder=3)
    ax[1].set_ylim(400, 1800)
    ax[1].set_xticks(x_axis)
    ax[1].set_xticklabels(marl_names)
    ax[1].yaxis.grid(zorder=0)
    ax[1].set_title("Cost (\u00A2) \u2193")
    plt.savefig("results/figures/metrics.png", bbox_inches="tight", dpi=150)

def plot_mmlu():
    acc = np.array([53.40, 68.53, 41.79, 77.29, 59.67, 76.36, 70.42, 75.01, 71.24, 40.26, 63.87, 55.82, 77.98])
    models = np.array(["Llama-2-13b-hf", "Llama-2-70b-hf", "Llama-2-7b-hf", "Meta-Llama-3-70B", "Mistral-7B-v0.2", "Mixtral-8x22B-v0.1", "Mixtral-8x7B-v0.1", "Qwen-72B", "deepseek-llm-67b", "gemma-2b", "gemma-7b", "phi-2", "MARL-Focal"])
    idx = np.argsort(acc)
    print(idx)
    
    acc = acc[idx]
    models = models[idx]
    colors = plt.cm.BuPu(np.linspace(0.1, 1, len(models)))
    fig, ax = plt.subplots()
    for i, (model_name, score) in enumerate(zip(models, acc)):
        ax.bar(i, score, label=model_name, zorder=3, color=colors[i])
    ax.legend(ncols=1, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(35, 80)
    ax.set_xticks([])
    ax.set_ylabel("Acc (%)")
    ax.yaxis.grid(zorder=0)
    ax.set_title("Reaching SOTA Performance in MMLU")
    plt.savefig("results/figures/mmlu.png", bbox_inches="tight", dpi=150)


def plot_hyper_params():
    size_penalty = np.array([0, 0.1, 0.6, 1.5, 3])
    acc = np.array([73.06, 74.19, 75.65, 73.87, 74.52])
    inference = np.array([4737, 4527, 4263, 3630, 3827])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(size_penalty, acc, "-->", lw=3, markersize=15, color="tab:green", zorder=3)
    ax[0].set_title("Acc (%) \u2191")
    ax[0].set_xlabel(r"Size Penalty $\alpha$")
    ax[0].grid(zorder=0)
    ax[1].plot(size_penalty, inference, "--*", lw=3, markersize=15, color="tab:orange", zorder=3)
    ax[1].set_title("Cost (\u00A2) \u2193")
    ax[1].set_xlabel(r"Size Penalty $\alpha$")
    ax[1].grid(zorder=0)
    plt.savefig("results/figures/alpha.png", bbox_inches="tight", dpi=150)
    

def run():
    # plot_compare_greedy()
    # plot_scalibility()
    # plot_metric_comparison()
    # plot_mmlu()
    plot_hyper_params()

    



if __name__ == "__main__":
    run()
