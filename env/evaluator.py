import sys
import os
import json
import re
import subprocess
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from config import openai_token



class Evaluator:
    def __init__(self):
        pass

    def evaluate_sample(self, prompt, output):
        pass


class EvaluateHelpfulness(Evaluator):
    def __init__(self):
        super().__init__()
        self.is_preamble_called = False
        self.ref_model_out_dict = []
        self.cur_dir = os.path.join(SCRIPT_DIR, "helpfulness_results")


    def evaluate_sample(self, prompt, output, idx):
        if not self.is_preamble_called:
            self.preamble_call_()
        
        match = re.search(r"### Instruction:\s*(.*)", prompt, re.DOTALL)
        prompt = match.group(1).strip() if match else prompt

        output = output.replace("### Instruction", "").replace("\n", "").strip()
        output = output.replace("###END", "")
        model_out_dict = [{
            "instruction": prompt,
            "output": output
        }]

        file_name = os.path.join(self.cur_dir, f"temp.json")
        with open(file_name, 'w') as json_file:
            json.dump(model_out_dict, json_file, indent=4)

        reference_out_dict = [self.ref_model_out_dict[idx]]
        ref_name = os.path.join(self.cur_dir, f"reference_model.json")
        with open(ref_name, 'w') as json_file:
            json.dump(reference_out_dict, json_file, indent=4)

        command_txt = f"alpaca_eval --model_outputs {self.cur_dir}/temp.json " + \
                      f"--reference_outputs {self.cur_dir}/reference_model.json --output_path {self.cur_dir}/out"
        print(command_txt)
        p = subprocess.Popen(command_txt, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        retval = p.wait()
        if retval==0:
            out_scores  = pd.read_csv(f"{self.cur_dir}/out/alpaca_eval_gpt4/leaderboard.csv")
            acc = 100.00 == out_scores["win_rate"].values[0]
        else:
            acc = False
        return int(acc)

        
    def preamble_call_(self):
        subprocess.run(["export", f"OPENAI_API_KEY={openai_token}"], shell=True)
        subprocess.run(["export", f"IS_ALPACA_EVAL_2=False"], shell=True)
        self.is_preamble_called = True

        print("loading reference model outputs")
        ref_name = os.path.join(self.cur_dir, f"text_davinci_003_outputs.json")
        with open(ref_name, 'r') as json_file:
            self.ref_model_out_dict = json.load(json_file)


if __name__ == "__main__":
    # p = subprocess.Popen('ls', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # for line in p.stdout.readlines():
    #     print(line)
    # retval = p.wait()
    evaluator = EvaluateHelpfulness()
    
    q = "Below is an instruction that describes a task. Write a response that appropriately completes the request." + \
        "\n\n### Instruction:\nWhat are the names of some famous actors that started their careers on Broadway?"
    
    ans = "Some famous actors that started their careers on Broadway include Denzel Washington, Meryl Streep, \
        James Earl Jones, Audra McDonald, Hugh Jackman, and Sally Field.###END"

    evaluator.evaluate_sample(q, ans, 0)




