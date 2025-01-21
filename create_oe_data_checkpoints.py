import os
import numpy as np

from data_generator.data_oe_loader import DataCreatorOE
from env.evaluator import (EvaluateHelpfulness, EvaluateSafety,
                            EvaluateTruthfulness)



def create_checkpoints():
    save_dir = os.path.join("results", "checkpoints", "open_ended")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    evalator_help = EvaluateHelpfulness()
    evalator_safety =  EvaluateSafety()
    evalator_truth = EvaluateTruthfulness()

    data_creator = DataCreatorOE(model_names="all")
    train_data, test_data = data_creator.create(train_num_samples=500, test_num_samples=500)

    for ds_name, data in zip(["train", "test"], [train_data, test_data]):
        scores = []
        for i, row in data.iterrows():
            row_arr = row.values
            prompt = row_arr[0]
            helpfulness_scores = [evalator_help.evaluate_sample(prompt, row_arr[j], i, ds_name) 
                                  for j in range(1, 4)]
            safety_scores = [evalator_safety.evaluate_sample(prompt, row_arr[j]) 
                             for j in range(1, 4)]
            truth_scores = [evalator_truth.evaluate_sample(prompt, row_arr[j]) 
                             for j in range(1, 4)]
            scores.append([helpfulness_scores, safety_scores, truth_scores])
        scores = np.array(scores)
        save_path = os.path.join(save_dir, f"{ds_name}_scores.npy")
        np.save(save_path, scores)


if __name__ == "__main__":
    create_checkpoints()


