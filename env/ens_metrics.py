import numpy as np
from statsmodels.stats import inter_rater as irr

from .diversity_stats import calc_generalized_div
from .ens_methods import voting


def calc_div_acc(solution, hist_data):
    comb_idx = solution.astype(bool)

    # select ensemble set
    set_bin_arr = hist_data["error_arr"][:, comb_idx]
    set_preds = hist_data["pred_arr"][:, comb_idx]
    label_arr = hist_data["label_arr"]

    # calc focal diversity of ensemble
    focal_div = 0
    ens_size = sum(solution)
    for focal_idx in range(ens_size):
        focal_arr = set_bin_arr[:, focal_idx]
        neg_idx = np.where(focal_arr == 0)[0]
        neg_samp_arr = set_bin_arr[neg_idx]
        focal_div += calc_generalized_div(neg_samp_arr)
    focal_div /= ens_size

    # calculate accuracy of ensemble
    ens_pred = voting(set_preds, method="plurality")
    ens_pred_flatten = ens_pred.flatten()
    acc_score = np.mean(label_arr == ens_pred_flatten)

    if len(label_arr) < 2:
        fleiss_kappa = 1
    else:
        dats, cats = irr.aggregate_raters(set_preds)
        fleiss_kappa = irr.fleiss_kappa(dats, method='fleiss')
    return focal_div, acc_score, fleiss_kappa

def fitness_function(solution, weights, hist_data, size_penalty=0):
    if sum(solution) < 2:
        score = -99
    else:
        focal_div, acc_score, kappa = calc_div_acc(solution, hist_data)
        score = focal_div * weights[0] + acc_score * weights[1] + kappa * weights[2]
        if size_penalty:
            score -= 0.1 * sum(solution)/len(solution)
    return score
