"""
This file runs free generation given a trained language model
"""
import re
import torch
import numpy as np
import pandas as pd
from itertools import product
from argparse import ArgumentParser
from tqdm import tqdm
import pickle
import sys
from pyprojroot import here

sys.path.extend(["../core", "./code/core"])
from utils import set_up_transformer, get_probability_from_transformer, PAD_TOKEN_ID, EQUALS_TOKEN_ID

def tokenize_vars(all_vars, tokenizer, args):
    var_tokens = {}
    for var in all_vars:
        tokens = tokenizer(var, return_tensors="pt").input_ids.to(args.device)
        var_tokens[var] = tokens.squeeze()

    return var_tokens

def get_next_var_probs(prefix, tokenizer, model, args, var_tokens):
    # generate up to the next equals sign
    token_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(args.device)
    outputs = model.generate(token_ids, return_dict_in_generate=True, output_scores=True,
                             max_new_tokens=2, pad_token_id=PAD_TOKEN_ID, eos_token_id=EQUALS_TOKEN_ID,
                             return_tensors="pt")
    # get the logits from the generation
    output_logits = outputs.scores
    # convert the relevant subset into probabilities
    vars_lst = list(var_tokens.keys())
    tokens_lst = list([var_tokens[t] for t in vars_lst])
    var_logits = []
    for token_vars, tokens in zip(vars_lst, tokens_lst):
        sum_logit = 0
        for i, token in enumerate(tokens):
            try:
                sum_logit += output_logits[i][0][token]
            except:
                print("tuple index out of range error!")
                print("output logits shape")
                print(output_logits.shape)
                print(f"i: {i}")
                print(f"token: {token}")
        var_logits.append(sum_logit)
    var_logits = torch.tensor(var_logits)

    # turn the logits into probabilities
    var_probs = torch.softmax(var_logits, dim=-1)

    return var_probs

def main(args):

    model, tokenizer = set_up_transformer(args.model_folder, device=args.device)
    nets = pickle.load(open(here(args.bayes_net_file), "rb"))
    net = nets[args.net_idx]
    net.from_pickle()
    all_vars = net.graph.nodes
    var_tokens = tokenize_vars(all_vars, tokenizer, args)

    df_pairs = pd.read_csv(here(f"data/training-data/selected-pairs/selected-pairs-net-{args.net_idx}.csv"))
    chosen_pairs = set()
    for index, row in df_pairs.iterrows():
        chosen_pairs.add((row["var1"], row["var2"]))
        chosen_pairs.add((row["var2"], row["var1"]))
        
    chosen_pairs = set(list(chosen_pairs))

    rows = []
    for target_var, condition_var in tqdm(chosen_pairs):
        if target_var == condition_var:
            continue

        for condition_val in (0, 1):
            target_probs, ns_intermediate, all_intermediate_vars, is_d_separating = [], [], set(), []
            for sample in range(args.num_samples):
                prefix = f"###\ntarget: {target_var}\n{condition_var}={condition_val}\n"
                generated_var = None
                target_prob = np.NaN
                intermediate_vars = set()
                for i in range(len(all_vars)):
                    var_probs = get_next_var_probs(prefix, tokenizer, model, args, var_tokens)
                    next_var = np.random.choice(all_vars, p=var_probs.numpy())
                    prefix += next_var + "="
                    if next_var == target_var:
                        target_prob = get_probability_from_transformer(prefix, tokenizer, model, args)
                        break
                    elif next_var != condition_var:
                        intermediate_vars.add(next_var)
                    prob = get_probability_from_transformer(prefix, tokenizer, model, args)
                    val = np.random.choice(2, p=[1 - prob, prob])
                    prefix += str(val) + "\n"
                if not np.isnan(target_prob):
                    target_probs.append(target_prob)
                    ns_intermediate.append(len(intermediate_vars))
                    is_d_separating.append(int(net.check_d_separated(condition_var, target_var, intermediate_vars)))
                    all_intermediate_vars |= intermediate_vars

            rows.append({
                "target_var": target_var,
                "condition_var": condition_var,
                "condition_val": condition_val,
                "prob": np.mean(target_probs),
                "prob_std": np.std(target_probs),
                "prop_valid": len(target_probs) / args.num_samples,
                "n_intermediate": np.mean(ns_intermediate),
                "prop_d_separating": np.mean(is_d_separating),
                "all_intermediate_vars": list(all_intermediate_vars)
            })

    df_free = pd.DataFrame(rows)
    return df_free

parser = ArgumentParser()
parser.add_argument("--model_folder", type=str)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--net_idx", type=int)
parser.add_argument("--bayes-net-file", type=str)
parser.add_argument("--num_samples", type=int, default=10)

if __name__ == "__main__":

    args = parser.parse_args()

    df_free = main(args)

    model_name = args.model_folder.split("/")[-1]
    df_free.to_csv(here(f"data/evaluation/free-gen-probabilities-{model_name}.csv"))
