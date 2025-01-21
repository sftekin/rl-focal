import os
import re
import openai
import json
import tqdm
import pandas as pd
from config import openai_token


def strip_instruct(prompt):
    match = re.search(r"### Instruction:\s*(.*)", prompt, re.DOTALL)
    prompt = match.group(1).strip() if match else prompt
    return prompt.replace("\n\n### Response:", "")


def run():
    # Set your OpenAI API key
    openai.api_key = openai_token

    data_df = pd.read_csv("data/open_ended/helpfulness/train/aligned/outputs_500.csv", index_col=0)

    out_dict = []
    for i, row in tqdm.tqdm(data_df.iterrows(), total=len(data_df)):
        prompt = row["prompts"]
        # Send the request to the API
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0.7,
            max_tokens=500
        )

        # Print the response
        generated_text = response.choices[0].text
        out_dict.append({
            "instruction": strip_instruct(prompt),
            "output": generated_text
        })

    ref_name = os.path.join(f"text_davinci_003_train.json")
    with open(ref_name, 'w') as json_file:
        json.dump(out_dict, json_file, indent=4)


if __name__ == "__main__":
    run()
