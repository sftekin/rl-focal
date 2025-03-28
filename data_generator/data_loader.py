import os
import sys
import re
import glob
import tqdm
import numpy as np
import pandas as pd
import pickle as pkl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from config import DATA_DIR
from data_generator.data_helper import (
    load_bbh_data, bbh_ds_names, norm_data, ds_names_dispatcher, load_method_dispatcher)


class DataCreator:
    def __init__(self, dataset_name, task_type="lang", model_names="all"):
        if task_type not in ["lang", "vision"]:
            raise RuntimeError("task type is not understood.")
        if task_type == "lang" and dataset_name not in ["gsm8k", "mmlu_hf", "bbh", "xsum", "gpqa", "musr"]:
            raise RuntimeError("input dataset can't found")
        elif task_type == "vision" and dataset_name not in ["miniimagenet", "cub", "fc100"]:
            raise RuntimeError("input dataset can't found")

        self.dataset_load_dict = {
            "mmlu_hf": self._load_mmlu_prob_and_label,
            "gsm8k": self._load_gsm8k_prob_and_label,
            "bbh": self._load_other_prob_and_label,
            "musr": self._load_other_prob_and_label,
            "gpqa": self._load_other_prob_and_label
        }

        self.task_type = task_type
        self.dataset_name = dataset_name
        self.model_names = model_names

        # some dataset specific arguments
        self.train_ratio = 0.8
    
    def _get_model_names(self, data_path):
        if self.model_names == "all":
            model_names = sorted([os.path.basename(fname) for fname in 
                                  glob.glob(os.path.join(data_path, "*")) 
                                  if os.path.isdir(fname)])
        else:
            model_names = self.model_names
        return model_names


    def load(self):
        if self.task_type == "lang":
            train_data_list, test_data_list, num_models, ds_names = (
                self.dataset_load_dict[self.dataset_name](
                    dataset_name=self.dataset_name
                )
            )
        if num_models == 0:
            raise RuntimeError
        
        for train_d, test_d, ds_name in zip(train_data_list, test_data_list, ds_names):
            yield train_d, test_d, num_models, ds_name


    def _load_mmlu_prob_and_label(self, **kwargs):
        data_path = os.path.join(DATA_DIR, "lang_datasets", "mmlu_hf")
        model_names = self._get_model_names(data_path)
        print(model_names)

        num_models = len(model_names)
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
        ds_len = len(data)
        train_size = int(ds_len * self.train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]

        return [train_data], [test_data], num_models, ["mmlu_hf"]

    def _load_gsm8k_prob_and_label(self, **kwargs):
        data_path = os.path.join(DATA_DIR, "lang_datasets", "gsm8k")
        model_names = self._get_model_names(data_path)
        print(model_names)
        num_models = len(model_names)

        train_data = self._load_gsm8k_dataset(data_path, model_names, dataset_name="train")
        test_data = self._load_gsm8k_dataset(data_path, model_names, dataset_name="test")

        return [train_data], [test_data], num_models, ["gsm8k"]

    def _load_other_prob_and_label(self, **kwargs):
        dataset_name = kwargs["dataset_name"]
        all_model_data = {
            "meta-llama__Llama-2-70b-chat-hf": {},
            "mistralai__Mixtral-8x7B-Instruct-v0.1" : {},
            "google__gemma-7b": {},
            "microsoft__phi-2": {},
            "mistralai__Mistral-7B-v0.1": {},
        }
        model_names = list(all_model_data.keys())
        sub_dir_names = ds_names_dispatcher[dataset_name]
        load_data_fun = load_method_dispatcher[dataset_name]

        temp_path = f"{dataset_name}_data.pkl"
        if os.path.exists(temp_path):
            with open(temp_path, "rb") as f:
                [train_data_list, test_data_list] = pkl.load(f)
        else:
            for mn in list(all_model_data.keys()):
                for ds_name in tqdm.tqdm(sub_dir_names):
                    x, y = load_data_fun(mn, ds_name)
                    all_model_data[mn][ds_name] = {
                        "data" : x,
                        "label": y
                    }
            train_data_list, test_data_list = [], []
            for ds_name in tqdm.tqdm(sub_dir_names):
                all_x, all_y = [], []
                for mn in all_model_data.keys():
                    x, y = all_model_data[mn][ds_name].values()
                    all_x.append(x)
                    all_y.append(y)
                all_x = np.concatenate(all_x, axis=1)

                try:
                    for y_idx in all_y:
                        for i in range(len(y_idx)):
                            assert(y_idx[i] == y[i])
                except AssertionError as e:
                    print(e)
                    print(f"skipping ... {ds_name}")

                concat_data = np.concatenate([all_x, y[:, None]], axis=1)
                train_size = int(len(all_x) * 0.8)
                train_data, test_data = concat_data[:train_size], concat_data[train_size:]
                train_data_list.append(train_data)
                test_data_list.append(test_data)

            with open(temp_path, "wb") as f:
                pkl.dump([train_data_list, test_data_list], f)

        return train_data_list, test_data_list, len(model_names), sub_dir_names

    def _load_gsm8k_dataset(self, data_path, model_names, num_samples=None,
                            num_runs=10, space_size=10, drop_non_exists=True, 
                            dataset_name="train"):
        # find number of samples for each model
        model_sample_count = []
        for model_n in model_names:
            results_dir = os.path.join(data_path, model_n, dataset_name)
            all_files_id = [int(fn.split("_")[1]) for fn in
                            os.listdir(results_dir) if "npy" in fn]
            model_sample_count.append(max(all_files_id))
        min_size = min(model_sample_count)

        if num_samples is None:
            num_samples = min_size
        elif num_samples > min_size:
            model_n = [model_n for i, model_n in enumerate(model_names)
                    if model_sample_count[i] < min_size]
            raise ValueError(f"{' '.join(model_n)} models have less then {num_samples}")

        # load the maximum and then truncate
        all_pred = []
        for i, model_n in enumerate(model_names):
            pred_path = os.path.join(data_path, model_n, dataset_name,
                                    f"run_{model_sample_count[i]}_predictions.npy")
            pred_arr = np.load(pred_path)[:num_samples, :num_runs]
            assert pred_arr.shape == (num_samples, num_runs)
            all_pred.append(pred_arr)
        all_pred_arr = np.concatenate(all_pred, axis=1)

        # extract probabilities for each model
        data_probs, solution_space = [], []
        for i in range(len(all_pred_arr)):
            sol_space = np.unique(all_pred_arr[i])
            probs_per_model = []
            for model_pred in all_pred:
                uni, counts = np.unique(model_pred[i], return_counts=True)
                count_dict = dict(zip(uni, counts))
                model_prob = []
                for j in sol_space:
                    if j in count_dict.keys():
                        model_prob.append(count_dict[j] / sum(counts))
                    else:
                        model_prob.append(0)
                probs_per_model.append(np.array(model_prob))
            # sort according to probabilities
            idx = np.argsort(np.sum(probs_per_model, axis=0))[::-1]
            probs_per_model = [prob[idx] for prob in probs_per_model]
            sol_space = sol_space[idx]
            data_probs.append(probs_per_model)
            solution_space.append(sol_space)

        # make each sample output to have the same space
        max_space_size = max([len(data[0]) for data in data_probs])
        if space_size < max_space_size:
            print("Truncating the space size")
        for i in range(len(data_probs)):
            if len(data_probs[i][0]) < space_size:
                pad_count = space_size - len(data_probs[i][0])
                pad_arr = np.zeros(pad_count)
                solution_space[i] = np.concatenate([solution_space[i], pad_arr])
                for j in range(len(data_probs[i])):
                    data_probs[i][j] = np.concatenate([data_probs[i][j], pad_arr])
            else:
                for j in range(len(data_probs[i])):
                    data_probs[i][j] = data_probs[i][j][:space_size]
                solution_space[i] = solution_space[i][:space_size]
        data_probs = np.stack(data_probs).reshape(num_samples, -1)
        solution_space = np.stack(solution_space)

        _, labels = self.load_gsm8k_raw_data(dataset_name=dataset_name)
        labels = labels[:num_samples]

        # drop episodes that are not in the solution space (do this only for training)
        x, y = [], []
        for i in range(len(labels)):
            if labels[i] in solution_space[i]:
                y.append(np.argwhere(solution_space[i] == labels[i])[0].item())
            else:
                y.append(np.nan)
            x.append(data_probs[i])
        y, x = np.array(y), np.array(x)
        data = np.concatenate([x, y[:, None]], axis=1)

        if drop_non_exists:
            idx = ~np.isnan(y)
            data = data[idx]

        return data

    def load_gsm8k_raw_data(self, dataset_name):
        data_path = os.path.join(DATA_DIR, "lang_datasets", "gsm8k", f"{dataset_name}.jsonl")
        data_df = pd.read_json(data_path, lines=True)

        questions, labels = [], []
        pattern = r'####\s*(\S+)'
        for i, row in data_df.iterrows():
            matches = re.findall(pattern, row.answer)
            labels.append(float(matches[0].replace(",", "")))
            questions.append(row.question)

        return questions, labels




if __name__ == "__main__":
    # m_names = ["Llama-2-13b-hf", "Llama-2-7b-hf", 
    #            "Mixtral-8x7B-v0.1", "gemma-7b", "Llama-2-70b-hf", 
    #            "Mistral-7B-Instruct-v0.2", "gemma-2b", "phi-2"]
    # f_data = load_mmlu_prob_and_label(m_names)
    datacreator = DataCreator(dataset_name="mmlu_hf")
    all_acc = []
    for train_data, test_data, num_models, ds_name in datacreator.load():
        print(train_data.shape, test_data.shape)
        space_size = (train_data.shape[1] - 1) // num_models

        novel_data = test_data[:, :-1]
        novel_label = test_data[:, -1]
        acc = []
        for i, novel_arr in enumerate(np.split(novel_data, num_models, axis=1)):
            pred = novel_arr.argmax(axis=1)
            acc.append(np.mean(novel_label == pred))
        acc = np.array(acc)
        all_acc.append(acc)
        print(ds_name, acc)
    print(np.mean(all_acc, axis=0))
            



