import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

from data_generator.data_loader import DataCreator


def run(args):
    datacreator = DataCreator(args.task_name, model_names=args.model_names, task_type="lang")
    for i, (train_data, test_data, num_models, ds_name) in enumerate(datacreator.load()):
        print()
    
    
if __name__ == "__main__":
    # Running with the same model pool as LLM-TOPLA.
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_names", type=str, default="all")
    parser.add_argument("--dataset_type", type=str, default="lang", choices=["lang", "vision"])
    parser.add_argument("--task_name", type=str, default="musr", choices=["gsm8k", "mmlu_hf", "bbh", "gpqa", "musr"])

    # select agent arguments
    parser.add_argument("--alpha", type=float, default=1.0, help="size penalty constant")
    parser.add_argument("--select_agent_epoch", type=int, default=15)
    parser.add_argument("--select_agent_gamma", type=float, default=0.99)
    parser.add_argument("--select_agent_clip_epsilon", type=float, default=0.2)
    parser.add_argument("--select_agent_lr", type=float, default=0.001)
    parser.add_argument("--focal_div_weight", type=float, default=0.5)
    parser.add_argument("--plurality_voting_weight", type=float, default=0.5)
    parser.add_argument("--fleiss_kappa_weight", type=float, default=0)
    parser.add_argument("--q_stats_weight", type=float, default=0)
    parser.add_argument("--corr_coef_weight", type=float, default=0)
    parser.add_argument("--binary_disagreement_weight", type=float, default=0)
    parser.add_argument("--kappa_stats_weight", type=float, default=0)
    parser.add_argument("--binary_entropy_weight", type=float, default=0)

    # ensemble agent arguments
    parser.add_argument("--ensemble_agent_epoch", type=int, default=15)
    parser.add_argument("--ens_agent_gamma", type=float, default=0.5)
    parser.add_argument("--ens_agent_clip_epsilon", type=float, default=0.2)
    parser.add_argument("--ens_agent_lr", type=float, default=0.001)

    # common argumentss for both agents
    parser.add_argument("--window_size", type=int, default=500)
    parser.add_argument("--max_tolerance", type=int, default=150)
    parser.add_argument("--update_freq", type=int, default=100)

    # Simulate command-line arguments
    if 'ipykernel' in sys.modules:
        sys.argv = ['notebook.py']  # Replace with desired arguments
    else:
        sys.argv = ['script.py']

    # Parse the arguments
    arguments = parser.parse_args()
    run(arguments)

