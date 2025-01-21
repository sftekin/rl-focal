import os
import sys
import glob
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from config import DATA_DIR


class DataCreatorOE:
    def __init__(self, model_names):
        self.model_names = model_names
        self.num_samples_for_each_task = {
            "train": {
                "helpfulness": 0,
                "safety":0,
                "truthfulness":0
            },
            "test": {
                "helpfulness": 0,
                "safety":0,
                "truthfulness":0
            }
            
        }

    def create(self, train_num_samples=500, test_num_samples=500):
        train_data = self._load_data_df(dataset_type="train", max_num_samples=train_num_samples)
        test_data = self._load_data_df(dataset_type="test", max_num_samples=test_num_samples)
        return train_data, test_data

    def _load_data_df(self, dataset_type="train", max_num_samples=500):
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
                questions = data_df["prompts"][:max_num_samples]
                outputs.append(data_df["outputs"][:max_num_samples])
            prompts.append(questions)
            data.append(pd.concat(outputs, axis=1))
            self.num_samples_for_each_task[dataset_type][task_name] = len(questions)

        data = pd.concat(data)
        prompts = pd.concat(prompts)
        data = pd.concat([prompts, data], axis=1)

        return data


if __name__ == "__main__":
    data_creator = DataCreatorOE(model_names="all")
    train_data, test_data = data_creator.create()


