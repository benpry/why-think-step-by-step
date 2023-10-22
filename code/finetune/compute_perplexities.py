"""
Compute the perplexity of a dataset under different language models.
"""
import os
import sys
import torch
import pandas as pd
from argparse import ArgumentParser
sys.path.extend(["../core", "./code/core"])
from utils import set_up_transformer
from pyprojroot import here

def compute_perplexity(model_folder, device='cuda:0'):

    command_str = f"python code/finetune/run_clm.py \
		--model_name_or_path {model_folder} \
                --do_eval \
                --validation_file data/training-data/samples/eval_samples_{model_name.replace('-', '_')}.csv \
		--per_device_eval_batch_size 3 \
		--output_dir {model_folder}"

    print(command_str)

    os.system(command_str)


parser = ArgumentParser()
parser.add_argument("--model_folder", type=str)
parser.add_argument("--base_arch", type=str)
parser.add_argument("--include_checkpoints", action="store_true", default=False)

if __name__ == "__main__":

    args = parser.parse_args()
    model_name = args.model_folder.split("/")[-1]

    compute_perplexity(args.model_folder)

    perps = []
    if args.include_checkpoints:
        for dir_name in os.listdir(args.model_folder):
            if "checkpoint" in dir_name:
                compute_perplexity(args.model_folder + "/" + dir_name)
