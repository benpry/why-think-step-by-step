"""
This file gets the pairs of variables that were eligible for holding out, but not chosen in the random sampling
"""
from argparse import ArgumentParser
import pandas as pd
from pyprojroot import here
import sys
sys.path.extend(["../core", "./code/core"])

parser = ArgumentParser()
parser.add_argument("--n_nets", type=int)
parser.add_argument("--n_pairs", type=int)

if __name__ == "__main__":

    args = parser.parse_args()

    mean_mi_rows = []

    for net_idx in range(args.n_nets):

        df_net = pd.read_csv(here(f"data/evaluation/mutual-informations/mutual-informations-net-{net_idx}.csv"))
        df_held_out = pd.read_csv(here(f"data/training-data/selected-pairs/selected-pairs-net-{net_idx}.csv"))

        # filter out adjacent pairs
        df_distant = df_net[df_net["distance"] > 1]

        df_largest_effects = df_distant.sort_values("mutual_information", ascending=False).head(2 * args.n_pairs)
        df_non_selected_pairs = df_largest_effects.merge(df_held_out,
                                                         on=("var1", "var2"),
                                                         how="left",
                                                         indicator=True)
        df_non_selected_pairs = df_non_selected_pairs[df_non_selected_pairs["_merge"] == "left_only"]
        df_non_selected_pairs = df_non_selected_pairs[["var1", "var2"]]

        df_non_selected_pairs.to_csv(here(f"data/training-data/non-selected-pairs/non-selected-pairs-net-{net_idx}.csv"), index=False)
