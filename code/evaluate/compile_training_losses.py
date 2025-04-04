"""
Compile loss on the training dataset
"""
import json
import pandas as pd
from argparse import ArgumentParser
from pyprojroot import here

parser = ArgumentParser()
parser.add_argument("--model_folder", type=str, required=True)
parser.add_argument("--base_arch", type=str, required=True)

if __name__ == "__main__":

    args = parser.parse_args()

    with open(f"{args.model_folder}/trainer_state.json", "r") as f:
        trainer_state = json.load(f)

    rows = []
    for checkpoint in trainer_state["log_history"]:
        if "loss" in checkpoint:
            rows.append({
                "n_steps": checkpoint["step"],
                "epoch": checkpoint["epoch"],
                "train_loss": checkpoint["loss"],
            })

    df = pd.DataFrame(rows)
    model_name = args.model_folder.split("/")[-1]
    df.to_csv(here(f"data/evaluation/base-model-{args.base_arch}/losses/train-losses-{model_name}.csv"), index=False)
