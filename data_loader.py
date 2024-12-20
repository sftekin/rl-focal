import os
import glob
import numpy as np
import pandas as pd

from config import DATA_DIR


class DataCreator:
    def __init__(self, dataset_name, task_type="lang", model_names="all"):
        if task_type not in ["lang", "vision"]:
            raise RuntimeError("task type is not understood.")

        if task_type == "lang" and dataset_name not in ["gsm8k", "mmlu_hf", "search_qa", "xsum"]:
            raise RuntimeError("input dataset can't found")
        elif task_type == "vision" and dataset_name not in ["miniimagenet", "cub", "fc100"]:
            raise RuntimeError("input dataset can't found")

        self.dataset_load_dict = {
            "mmlu_hf": self._load_mmlu_prob_and_label
        }

        self.model_names = model_names
        self.data = None
        if task_type == "lang":
            self.data = self.dataset_load_dict[dataset_name]()


    def _load_mmlu_prob_and_label(self):
        data_path = os.path.join(DATA_DIR, "lang_datasets", "mmlu_hf")

        if self.model_names == "all":
            model_names = [os.path.basename(fname) for fname in 
                           glob.glob(os.path.join(data_path, "*"))]
        else:
            model_names = self.model_names

        data_df_names = [os.path.basename(fname) for fname in
                        glob.glob(f"{data_path}/{model_names[0]}/*.csv")]
        data_df_names = sorted(data_df_names)

        data = []
        choices = ["A", "B", "C", "D"]
        for df_name in data_df_names:
            df_pred, df_label = [], []
            for m_name in model_names:
                dpath = f"{data_path}/{m_name}/{df_name}"
                data_df = pd.read_csv(dpath)
                a = data_df["prediction"].apply(lambda x: np.array(eval(x)))
                b = data_df["label"].apply(lambda x: choices.index(x))
                df_pred.append(np.exp(np.stack(a.values)))
                df_label.append(b.values)
            df_pred = np.concatenate(df_pred, axis=1)
            data.append(np.concatenate([df_pred, df_label[0][:, None]], axis=1))
        data = np.concatenate(data)
        return data


if __name__ == "__main__":
    # m_names = ["Llama-2-13b-hf", "Llama-2-7b-hf", 
    #            "Mixtral-8x7B-v0.1", "gemma-7b", "Llama-2-70b-hf", 
    #            "Mistral-7B-Instruct-v0.2", "gemma-2b", "phi-2"]
    # f_data = load_mmlu_prob_and_label(m_names)
    datacreator = DataCreator(dataset_name="mmlu_hf")

    print(len(datacreator.data))


