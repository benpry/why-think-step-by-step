"""
This file computes probabilities with "fixed generation" where we don't generate any individual values
"""
import torch
from pyprojroot import here
from itertools import product
from argparse import ArgumentParser
import pickle
import pandas as pd
import sys

sys.path.extend(["../core", "./core", "./code/core"])
from utils import set_up_transformer, get_probability_from_transformer

def main(args):
    model, tokenizer = set_up_transformer(args.model_folder, device=args.device)

    net_idx = args.net_idx
    nets = pickle.load(open(here(args.bayes_net_file), "rb"))
    net = nets[net_idx]
    net.from_pickle()

    if args.only_selected_vars:
        df_pairs = pd.read_csv(here(f"data/training-data/selected-pairs/selected-pairs-net-{args.net_idx}.csv"))
        chosen_pairs = set()
        for index, row in df_pairs.iterrows():
            chosen_pairs.add((row["var1"], row["var2"]))
            chosen_pairs.add((row["var2"], row["var1"]))
    else:
        all_vars = net.graph.nodes
        chosen_pairs = set(product(all_vars, repeat=2))

    rows = []
    for target, condition in chosen_pairs:
        for condition_val in (0, 1):
            if target == condition:
                continue

            prefix = f"###\ntarget: {target}\n{condition}={condition_val}\n{target}="

            prob = get_probability_from_transformer(prefix, tokenizer, model, args)

            rows.append({
                "target_var": target,
                "condition_var": condition,
                "condition_val": condition_val,
                "prob": prob
            })

    df = pd.DataFrame(rows)
    return df


parser = ArgumentParser()
parser.add_argument("--model_folder", type=str)
parser.add_argument("--net_idx", type=int)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--bayes-net-file", type=str)
parser.add_argument("--only_selected_vars", action="store_true")
parser.add_argument("--base_model_name", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    df = main(args)
    model_name = args.model_folder.split("/")[-1]
    df.to_csv(here(f"data/evaluation/base-model-{args.base_model_name}/fixed-gen-probabilities-{model_name}.csv"))
