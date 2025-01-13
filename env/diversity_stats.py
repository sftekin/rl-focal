import numpy as np
import pandas as pd


def pairwise_stat(fir_arr, sec_arr):
    def only_one_corr(a, b):
        xor_out = np.logical_xor(a, b)
        and_out = np.logical_and(xor_out, a)
        return and_out
    n_11 = ((fir_arr + sec_arr) > 1).sum()
    n_00 = ((fir_arr + sec_arr) == 0).sum()
    n_10 = only_one_corr(fir_arr, sec_arr).sum()
    n_01 = only_one_corr(sec_arr, fir_arr).sum()
    return n_11, n_00, n_10, n_01


def q_stat(n_11, n_00, n_10, n_01):
    nom = (n_11 * n_00) - (n_01 * n_10)
    den = (n_11 * n_00) + (n_01 * n_10)
    return nom / den


def ro_corr(n_11, n_00, n_10, n_01):
    nom = (n_11 * n_00) - (n_01 * n_10)
    den = np.sqrt((n_11 + n_10) * (n_01 + n_00)
                  * (n_11 + n_01) * (n_10 + n_00))
    return nom / den


def bin_dis(n_11, n_00, n_10, n_01):
    nom = n_01 + n_10
    den = n_11 + n_00 + n_10 + n_01
    return nom / den


def kapp_stat(n_11, n_00, n_10, n_01):
    nom = (n_11 * n_00) - (n_01 * n_10)
    den = ((n_11 + n_10) * (n_01 + n_00)) + ((n_11 + n_01) * (n_10 + n_00))
    return 2 * nom / den


def calc_stat(fir_arr, sec_arr, stat_method):
    all_stat = pairwise_stat(fir_arr, sec_arr)
    stat_val = stat_method(*all_stat)
    return stat_val


def calc_stat_matrices(errors):
    stat_dict = {
        "q_statistics": q_stat,
        "correlation_co-efficiency": ro_corr,
        "binary_disagreement": bin_dis,
        "kappa_statistics": kapp_stat
    }
    model_names = list(errors.keys())
    stat_matrices = {}
    for stat_name, stat_method in stat_dict.items():
        stat_arr = np.ones((len(model_names), len(model_names)))
        for i, f_name in enumerate(model_names):
            for j, s_name in enumerate(model_names):
                stat_arr[i, j] = calc_stat(fir_arr=errors[f_name],
                                           sec_arr=errors[s_name],
                                           stat_method=stat_method)
        stat_df = pd.DataFrame(stat_arr, columns=model_names, index=model_names)
        stat_matrices[stat_name] = stat_df
    return stat_matrices


def calc_generalized_div(binary_preds):
    n_samples, model_size = binary_preds.shape

    inv_arr = 1 - binary_preds
    err_count = inv_arr.sum(axis=1)
    occurrence, freq = np.unique(err_count, return_counts=True)
    pi = freq / n_samples

    if model_size > len(pi):
        temp_pi = [pi[occurrence.tolist().index(occ)] if occ in occurrence else 0
                   for i, occ in enumerate(range(1, model_size + 1))]
        pi = temp_pi

    p_1 = 0
    p_2 = 0
    for i in range(1, model_size + 1):
        idx = i - 1
        p_1 += i * pi[idx] / model_size
        p_2 += i * (i - 1) * pi[idx] / (model_size * (model_size - 1))

    if p_1 == 0:
        gd = 1
    else:
        gd = 1 - (p_2 / p_1)

    return gd

