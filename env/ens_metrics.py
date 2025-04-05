import numpy as np
from statsmodels.stats import inter_rater as irr

from .diversity_stats import calc_generalized_div, calc_pairwise_arr, calc_stat_matrices, calc_binary_entropy
from .ens_methods import voting


def calc_div_acc(solution, hist_data, weights):
    comb_idx = solution.astype(bool)
    score = 0

    # select ensemble set
    set_bin_arr = hist_data["error_arr"][:, comb_idx]
    set_preds = hist_data["pred_arr"][:, comb_idx]
    label_arr = hist_data["label_arr"]

    # calc focal diversity of ensemble
    if weights[0] > 0:
        focal_div = 0
        ens_size = sum(solution)
        for focal_idx in range(ens_size):
            focal_arr = set_bin_arr[:, focal_idx]
            neg_idx = np.where(focal_arr == 0)[0]
            neg_samp_arr = set_bin_arr[neg_idx]
            focal_div += calc_generalized_div(neg_samp_arr)
        focal_div /= ens_size
        score += weights[0] * focal_div
    
    # calculate voting accuracy of ensemble
    if weights[1] > 0:
        ens_pred = voting(set_preds, method="plurality")
        ens_pred_flatten = ens_pred.flatten()
        acc_score = np.mean(label_arr == ens_pred_flatten)
        score += weights[1] * acc_score

    # calculate fleiss kappa of ensemble
    if weights[2] > 0:
        if len(label_arr) < 2:
            fleiss_kappa = 1
        else:
            dats, cats = irr.aggregate_raters(set_preds)
            fleiss_kappa = irr.fleiss_kappa(dats, method='fleiss')
        score += weights[2] * fleiss_kappa

    comb = np.argwhere(solution).squeeze()
    error_dict = {str(i):hist_data["error_arr"][:, i] for i in range(len(comb_idx))}
    if sum(weights[3:]) > 0:
        stat_df = calc_stat_matrices(errors=error_dict)
        # calculate Q-statistics
        if weights[3] > 0:
            val = calc_pairwise_arr(stat_df, comb, "q_statistics")
            score += weights[3] * val
        # calculate Correlation-co-Efficiency
        if weights[4] > 0:
            val = calc_pairwise_arr(stat_df, comb, "correlation_co-efficiency")
            score += weights[4] * val
        # calculate Binary-Disagreement
        if weights[5] > 0:
            val = calc_pairwise_arr(stat_df, comb, "binary_disagreement")
            score += weights[5] * val
        # calculate Kappa-statistics
        if weights[6] > 0:
            val = calc_pairwise_arr(stat_df, comb, "kappa_statistics")
            score += weights[6] * val
        # calculate binary entropy
        if weights[7] > 0:
            val = calc_binary_entropy(set_bin_arr.flatten())
            score += weights[7] * val    

    return score

def fitness_function(solution, weights, hist_data):
    if sum(solution) < 2:
        score = -99
    else:
        score = calc_div_acc(solution, hist_data, weights)
    return score
