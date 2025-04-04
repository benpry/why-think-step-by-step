"""
Compile perplexities on the validation set for each model and checkpoint.
"""
import os
import json
import pandas as pd
from argparse import ArgumentParser
from pyprojroot import here

parser = ArgumentParser()
parser.add_argument("--model_folder", type=str, required=True)
parser.add_argument("--base_arch", type=str, required=True)

if __name__ == "__main__":

    args = parser.parse_args()
    model_name = args.model_folder.split("/")[-1]
    root_dir = args.model_folder

    all_eval_results = []
    for checkpoint_dir in os.listdir(root_dir):
        if checkpoint_dir.startswith("checkpoint"):
            checkpoint_num = checkpoint_dir.split("-")[-1]
            eval_results = json.load(open(os.path.join(root_dir, checkpoint_dir, "eval_results.json")))
            all_eval_results.append({
                "checkpoint_num": checkpoint_num,
                "perplexity": eval_results["perplexity"],
                "eval_loss": eval_results["eval_loss"]
            })

    df_eval_results = pd.DataFrame(all_eval_results)
    df_eval_results.to_csv(here(f"data/evaluation/base-model-{args.base_arch}/eval-results/eval-results-{model_name}.csv"), index=False)
