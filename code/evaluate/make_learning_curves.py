"""
This file creates "learning curves" for each checkpoint of a given model using fixed and free generation
"""
import os
from glob import glob
import pandas as pd
import re
from argparse import ArgumentParser
from free_generation_probabilities import main as get_free_generation_probabilities
from fixed_generation_probabilities import main as get_direct_prediction_probabilities
from pyprojroot import here
from copy import deepcopy

parser = ArgumentParser()
parser.add_argument("--model_folder", type=str)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--net_idx", type=int)
parser.add_argument("--bayes-net-file", type=str)
parser.add_argument("--num_samples", type=int, default=10)
if __name__ == "__main__":

    args = parser.parse_args()
    df_true = pd.read_csv(here(f"data/evaluation/true-probs/true-probabilities-net-{args.net_idx}.csv"))
    df_true["condition_val"] = df_true["condition_val"].apply(int)
    df_true["type"] = "true"

    checkpoints = glob(os.path.join(args.model_folder, "checkpoint-*"))
    checkpoints = sorted(checkpoints, key=lambda x: int(re.search(r"checkpoint-(\d+)", x).groups()[0]))

    df_all = pd.DataFrame()
    for i, checkpoint in enumerate(checkpoints):

        print("Processing checkpoint {} of {}".format(i+1, len(checkpoints)))

        checkpoint_args = deepcopy(args)
        checkpoint_args.model_folder = checkpoint
        checkpoint_args.only_selected_vars = True

        # Get free generation probabilities
        df_free = get_free_generation_probabilities(checkpoint_args)
        df_free["type"] = "free"

        # Get direct prediction probabilities
        df_direct = get_direct_prediction_probabilities(checkpoint_args)
        df_direct["type"] = "direct"

        df_all = pd.concat([df_all, df_free, df_direct])

    model_name = args.model_folder.split("/")[-1]
    df_all.to_csv(here(f"data/evaluation/learning-curves/learning-curves-{model_name}.csv"))
