"""
This file queries the model using the scaffolded generation order
"""
import torch
from transformers import pipeline
from omegaconf import DictConfig
from pyprojroot import here
from itertools import product
from argparse import ArgumentParser
from ast import literal_eval
from random import random
from tqdm import tqdm
import pickle
import pandas as pd
import sys
import json

sys.path.extend(["../core", "./core", "./code/core"])
from utils import set_up_transformer, get_probability_from_transformer

def generate_additional_variable(prefix, variable, tokenizer, model, args):
    """
    Generate the specified new variable, starting from the existing prefix
    """
    full_prefix = prefix + f"{variable}="
    prob = get_probability_from_transformer(full_prefix, tokenizer, model, args)
    new_val = 1 if random() < prob else 0
    new_str = f"{full_prefix}{new_val}\n"

    return new_str

parser = ArgumentParser()
parser.add_argument("--model_folder", type=str)
parser.add_argument("--scaffold_file", type=str)
parser.add_argument("--net_idx", type=int)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--bayes-net-file", type=str)
parser.add_argument("--negative", action="store_true", default=False)
parser.add_argument("--base_model_name", type=str)

if __name__ == "__main__":

    args = parser.parse_args()

    model, tokenizer = set_up_transformer(args.model_folder, device=args.device)
    net_idx = args.net_idx
    nets = pickle.load(open(here(args.bayes_net_file), "rb"))
    net = nets[net_idx]
    net.from_pickle()
    all_vars = net.graph.nodes

    if args.negative:
        df_scaffolds = pd.read_csv(f"data/scaffolds/negative-scaffolds-net-{args.net_idx}.csv")
    else:
        df_scaffolds = pd.read_csv(f"data/scaffolds/scaffolds-net-{args.net_idx}.csv")

    rows = []
    for index, row in tqdm(list(df_scaffolds.iterrows())):
        target, condition, condition_val = row["target_var"], row["condition_var"], row["condition_val"]
        scaffold_order = literal_eval(row["scaffold_order"])
        for sample in range(args.num_samples):
            if target == condition:
                continue

            prefix = f"###\ntarget: {target}\n{condition}={int(condition_val)}\n"
            for scaffold_var in scaffold_order:
                prefix = generate_additional_variable(prefix, scaffold_var, tokenizer, model, args)

            # Add the target variable to the prefix to get the target variable
            prefix += f"{target}="
            final_prob = get_probability_from_transformer(prefix, tokenizer, model, args)

            rows.append({
                "target_var": target,
                "condition_var": condition,
                "condition_val": condition_val,
                "prob": final_prob
            })

    model_name = args.model_folder.split("/")[-1]
    # save all the generated probabilities -- useful for computing standard deviations and such
    df = pd.DataFrame(rows)
    if args.negative:
        df.to_csv(here(f"data/evaluation/negative-scaffolded-gen-all-probabilities-{model_name}.csv"))
    else:
        df.to_csv(here(f"data/evaluation/scaffolded-gen-all-probabilities-{model_name}.csv"))

    df_summary = df.groupby(["target_var", "condition_var", "condition_val"]).agg({"prob": ["mean", "std"]}).reset_index()
    df_summary.columns = df_summary.columns.map(''.join)
    df_summary = df_summary.rename(columns={"probmean": "prob", "probstd": "prob_std"})
    if args.negative:
        df_summary.to_csv(here(f"data/evaluation/base-model-{args.base_model_name}/negative-scaffolded-gen-probabilities-{model_name}-{args.num_samples}samples.csv"))
    else:
        df_summary.to_csv(here(f"data/evaluation/base-model-{args.base_model_name}/scaffolded-gen-probabilities-{model_name}-{args.num_samples}samples.csv"))
