import os
os.environ['HF_HOME'] = "~/scratch/hf-cache"
import pickle as pkl
import numpy as np
import tqdm
import argparse

import pandas as pd
import re
import json

from transformers import pipeline


def load_gsm8k_data(dataset_name):
    data_path = os.path.join("raw_data", "gsm8k", f"{dataset_name}.jsonl")
    data_df = pd.read_json(data_path, lines=True)

    questions, labels = [], []
    pattern = r'####\s*(\S+)'
    for i, row in data_df.iterrows():
        matches = re.findall(pattern, row.answer)
        labels.append(float(matches[0].replace(",", "")))
        questions.append(row.question)

    return questions, labels


def get_prompt_template(task_name, n_shot):
    template_txt = "{prompt}"
    # temp_txt = [INST] {prompt} [/INST]

    prompt = ""
    if n_shot > 0:
        with open(os.path.join("raw_data", "few_shot.json"), "r") as f:
            temp_dict = json.load(f)
        examples = temp_dict[task_name]["examples"]
        examples = examples[:n_shot]
        prompt = temp_dict[task_name]["prompt"]
        prompt = prompt.replace("{examples}", "\n ".join(examples))
        # prompt = "example1, example2, ... {question}"

    template_txt = template_txt.replace("{prompt}", prompt)

    return template_txt


def main(args):
    
    pipe = pipeline("text-generation", model=f"microsoft/{args.llm_name}")

    questions, labels = load_gsm8k_data(dataset_name=args.dataset_name)
    prompt_template = get_prompt_template(task_name="gsm8k",
                                            n_shot=5)


    prompts = []
    for question in questions:
        p = prompt_template.replace("{question}", question)
        prompts.append(p)

    prompts = []
    for question in questions:
        p = prompt_template.replace("{question}", question)
        prompts.append(p)

    load_last_save = 1
    sample_per_pass = 5
    save_freq = 100
    save_dir = os.path.join(f"data/gsm8k/phi-4/{args.dataset_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        load_last_save = 0

    if load_last_save == 1 and os.listdir(save_dir):
        all_files_id = [int(fn.split("_")[1]) for fn in
                        os.listdir(save_dir) if "npy" in fn]
        id_max = max(all_files_id)
        arr_path = os.path.join(save_dir, f"run_{id_max}_predictions.npy")
        outputs_path = os.path.join(save_dir, f"run_{id_max}_outputs.pkl")
        predictions = np.load(arr_path).tolist()
        with open(outputs_path, "rb") as f:
            outputs = pkl.load(f)
        start_sample = id_max + 1
        prompts = prompts[start_sample:]
        print(f"Resuming at {id_max}th step")
    else:
        predictions, outputs = [], []
        start_sample = 0

    answer_pattern_2 = r"[-+]?\d*\.?\d+"
    count = start_sample
    for in_prompt in tqdm.tqdm(prompts):
        inner_pred, inner_output = [], []
        for j in range(10 // sample_per_pass):
            messages = [
                {"role": "user", "content": "Answer only the last question don't repeat the question" + in_prompt},
            ]
            out = pipe(
                messages,
                max_new_tokens=512,
                temperature=0.6,
                num_return_sequences=sample_per_pass,
                do_sample=True, 
                # top_p=0.95,
                # stop_sequences=["</s>", "####"],   # if supported by the model tokenizer)
            )

            for choice in out:
                answer_txt = choice["generated_text"][1]["content"]
                # print(answer_txt)
                matches = re.findall(answer_pattern_2, answer_txt)
                if len(matches) > 0:
                    inner_pred.append(float(matches[-1]))
                else:
                    inner_pred.append(np.nan)
                inner_output.append(answer_txt)

        predictions.append(np.array(inner_pred))
        outputs.append(inner_output)

        if count % save_freq == 0 or count == (len(prompts) - 1):
            save_pred_arr = np.array(predictions)
            arr_path = os.path.join(save_dir, f"run_{count}_predictions.npy")
            outputs_path = os.path.join(save_dir, f"run_{count}_outputs.pkl")
            np.save(arr_path, save_pred_arr)
            with open(outputs_path, "wb") as f:
                pkl.dump(outputs, f)
        count += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM inference')
    parser.add_argument('--task_name', default="gsm8k", type=str)
    parser.add_argument('--llm_name', default="phi-4",
                        choices=["phi-4"])
    parser.add_argument("--dataset_name", default="train", choices=["train", "test"])
    parser.add_argument("--num_samples", default=None, type=int)
    parser.add_argument("--n_shot", default=5, type=int)
    parser.add_argument("--sample_per_pass", default=10, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--load_last_save", default=1, type=int, choices=[0, 1])
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=40, type=int)
    arguments = parser.parse_args()
    main(arguments)
