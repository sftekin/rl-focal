import os
import glob
import numpy as np
import pandas as pd

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, "data")


def load_mmlu_prob_and_label(model_names):
    data_path = os.path.join(DATA_DIR, "lang_datasets", "mmlu_hf")
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
    m_names = ["Llama-2-13b-hf", "Llama-2-7b-hf", 
               "Mixtral-8x7B-v0.1", "gemma-7b", "Llama-2-70b-hf", 
               "Mistral-7B-Instruct-v0.2", "gemma-2b", "phi-2"]
    f_data = load_mmlu_prob_and_label(m_names)
    print(len(f_data))


