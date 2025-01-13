import numpy as np
from datasets import load_dataset
from config import hf_token

bbh_ds_names = ["boolean_expressions",  "causal_judgement",  "date_understanding",  "disambiguation_qa",
                "formal_fallacies",  "geometric_shapes",  "hyperbaton",  "logical_deduction_five_objects",
                "logical_deduction_seven_objects",  "logical_deduction_three_objects",  "movie_recommendation",
                "navigate",  "object_counting",  "penguins_in_a_table",  "reasoning_about_colored_objects",
                "ruin_names",  "salient_translation_error_detection",  "snarks",  "sports_understanding",
                "temporal_sequences", "tracking_shuffled_objects_five_objects",  
                "tracking_shuffled_objects_seven_objects",  "tracking_shuffled_objects_three_objects",  "web_of_lies"]



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
