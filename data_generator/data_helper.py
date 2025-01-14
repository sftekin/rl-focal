import numpy as np
import torch
from datasets import load_dataset
from config import hf_token
import torch.nn.functional as F


bbh_ds_names = ["boolean_expressions",  "causal_judgement",  "date_understanding",  "disambiguation_qa",
                "formal_fallacies",  "geometric_shapes",  "hyperbaton",  "logical_deduction_five_objects",
                "logical_deduction_seven_objects",  "logical_deduction_three_objects",  "movie_recommendation",
                "navigate",  "object_counting",  "penguins_in_a_table",  "reasoning_about_colored_objects",
                "ruin_names",  "salient_translation_error_detection",  "snarks",  "sports_understanding",
                "temporal_sequences", "tracking_shuffled_objects_five_objects",  
                "tracking_shuffled_objects_seven_objects",  "tracking_shuffled_objects_three_objects",  "web_of_lies"]
gpqa_ds_names = ["main"]
musr_ds_names = ["murder_mysteries", "object_placements", "team_allocation"]


def load_bbh_data(model_name, ds_name):
    data = load_dataset(
        f"HuggingFaceEvalInternal/{model_name}-details-private",
        name=f"{model_name}__leaderboard_bbh_{ds_name}",
        split="latest",
        token=hf_token)

    idx = np.argsort(data["doc_id"])
    probs = np.array(data["resps"]).squeeze()[:, :, 0].astype(float)
    labels = np.array([data["doc"][i]["target"] for i in range(len(probs))])
    choices = sorted(list(set(labels)))
    labels_list = [choices.index(lbl) for lbl in labels]
    label_arr = np.array(labels_list).astype(int)

    return probs[idx], label_arr[idx]


def load_gpqa_data(model_name, ds_name):
    data = load_dataset(
    f"HuggingFaceEvalInternal/{model_name}-details-private",
    name=f"{model_name}__leaderboard_gpqa_{ds_name}",
    split="latest",
    token=hf_token)

    idx = np.argsort(data["doc_id"])
    probs = np.array(data["resps"]).squeeze()[:, :, 0].astype(float)
    probs = F.softmax(torch.tensor(probs), dim=-1).numpy()
    labels = np.array([data["doc"][i]["answer"] for i in range(len(probs))])

    lbl_index = []
    for lbl in labels:
        lbl_index.append(["(A)", "(B)", "(C)", "(D)"].index(lbl))
    labels = np.array(lbl_index)

    return probs[idx], labels[idx]


def load_musr_data(model_name, ds_name):
    data = load_dataset(
    f"HuggingFaceEvalInternal/{model_name}-details-private",
    name=f"{model_name}__leaderboard_musr_{ds_name}",
    split="latest",
    token=hf_token)

    idx = np.argsort(data["doc_id"])
    labels = np.array([data["doc"][i]["answer_index"] for i in range(len(idx))])
    
    if ds_name == "object_placements":
        prob_pad = []
        for rep in data["resps"]:
            responses = np.array(rep).squeeze()[:, 0].astype(float)
            probs = F.softmax(torch.tensor(responses), dim=-1)
            num_zeros = 5 - len(probs)
            if num_zeros > 0:
                probs = torch.cat([probs, torch.zeros(num_zeros)])
            prob_pad.append(probs)
        probs = torch.stack(prob_pad)
    else:
        probs = np.array(data["resps"]).squeeze()[:, :, 0].astype(float)
        probs = F.softmax(torch.tensor(probs), dim=-1)
    probs = probs.numpy()
    # print(probs.shape, idx.shape, labels.shape)

    return probs[idx], labels[idx]


def norm_data(data):
	# Calculate min and max values
	min_val = np.min(data)
	max_val = np.max(data)

	# Normalize the array
	normalized_data = (data - min_val) / (max_val - min_val)
	return normalized_data


ds_names_dispatcher = {
    "bbh": bbh_ds_names,
    "gpqa": gpqa_ds_names,
    "musr": musr_ds_names
}


load_method_dispatcher = {
    "bbh": load_bbh_data,
    "gpqa": load_gpqa_data,
    "musr": load_musr_data
}
