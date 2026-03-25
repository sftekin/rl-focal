import os
import numpy as np
import tqdm

from data_generator.data_oe_loader import DataCreatorOE
from env.evaluator import (EvaluateHelpfulness, EvaluateSafety,
                            EvaluateTruthfulness)


def save_checkpoint(save_dir, ds_name, scores, idx):
    scores = np.array(scores)
    save_path = os.path.join(save_dir, f"{ds_name}_scores_{idx}.npy")
    np.save(save_path, scores)



def create_checkpoints():
    save_dir = os.path.join("results", "checkpoints", "open_ended")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    evalator_help = EvaluateHelpfulness()
    evalator_safety =  EvaluateSafety()
    evalator_truth = EvaluateTruthfulness()

    data_creator = DataCreatorOE(model_names="all")
    train_data, test_data = data_creator.create(train_num_samples=500, test_num_samples=500)

    test_data.index = np.arange((len(test_data)))
    train_data.index = np.arange((len(train_data)))

    for ds_name, data in zip(["train", "test"], [train_data, test_data]):
        if ds_name != "test":
            continue
        scores = []
        for i, row in tqdm.tqdm(data.iterrows(), total=len(data)):
            row_arr = row.values
            prompt = row_arr[0]

            if i < 500:
                ep_scores = [evalator_help.evaluate_sample(prompt, row_arr[j], i, ds_name) 
                                    for j in range(1, 4)]
                print("helpfulness", ep_scores)
            elif i < 1000:
                ep_scores = [evalator_safety.evaluate_sample(prompt, row_arr[j]) 
                                for j in range(1, 4)]
                print("safety", ep_scores)
            else:
                ep_scores = [evalator_truth.evaluate_sample(prompt, row_arr[j]) 
                                for j in range(1, 4)]
                print("truthfulness", ep_scores)
            
            scores.append(ep_scores)
            if i % 50 == 0:
                save_checkpoint(save_dir, ds_name, scores, i)
        save_checkpoint(save_dir, ds_name, scores, "end")

if __name__ == "__main__":
    create_checkpoints()


