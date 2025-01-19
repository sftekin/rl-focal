import os
import sys
import glob
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from config import DATA_DIR


class DataCreatorOE:
    def __init__(self, model_names, max_num_samples=500):
        self.max_num_samples = max_num_samples
        self.model_names = model_names

    def create(self):
        train_data = self._load_data_df(dataset_type="train")
        test_data = self._load_data_df(dataset_type="test")
        return train_data, test_data

    def _load_data_df(self, dataset_type="train"):
        data_path = os.path.join(DATA_DIR, "open_ended")
        
        data, prompts = [], []
        for task_name in ["helpfulness", "safety", "truthfulness"]:
            if self.model_names == "all":
                model_names = ["helpfulness", "safety", "truthfulness"]
            else:
                model_names = self.model_names
            questions, outputs = [], []
            for mn in model_names:
                if mn == task_name:
                    dir_name = "aligned"
                else:
                    dir_name = f"cross_{mn}"
                dir_path = os.path.join(data_path, task_name,
                                        dataset_type, dir_name)
                if os.path.exists(os.path.join(dir_path, "outputs_500.csv")):
                    data_df = pd.read_csv(os.path.join(dir_path, "outputs_500.csv"))
                else:
                    data_df = pd.read_csv(os.path.join(dir_path, "outputs_final.csv"))
                questions = data_df["prompts"][:self.max_num_samples]
                outputs.append(data_df["outputs"][:self.max_num_samples])
            prompts.append(questions)
            data.append(pd.concat(outputs, axis=1))
        data = pd.concat(data)
        prompts = pd.concat(prompts)
        data = pd.concat([prompts, data], axis=1)

        return data


if __name__ == "__main__":
    data_creator = DataCreatorOE(model_names="all")
    train_data, test_data = data_creator.create()


