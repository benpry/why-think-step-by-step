"""
Utilities that help with various tasks.
"""
import numpy as np
from pomegranate import BayesianNetwork as PomBN
from pgmpy.models import BayesianNetwork as PgmBN
from pgmpy.factors.discrete import TabularCPD
from transformers import AutoModelForCausalLM, GPT2TokenizerFast, AutoConfig, PretrainedConfig, pipeline
import torch
from random import random

# Constants for tokenizer
ZERO_TOKEN_ID = 15
ONE_TOKEN_ID = 16
PAD_TOKEN_ID = 1023
EQUALS_TOKEN_ID = 28

def set_up_transformer(model_folder, device="cuda:0", return_dict=True):
    """
    Read the transformer and tokenizer from a given folder
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model_folder)
    model_config = AutoConfig.from_pretrained(f"{model_folder}/config.json")
    model = AutoModelForCausalLM.from_pretrained(f"{model_folder}/pytorch_model.bin", 
        config=model_config).to(device)

    return model, tokenizer


def get_probability_from_transformer(prefix, tokenizer, model, args):
    """
    Get the probability of 1 given a particular prefix with a tokenizer and model
    """
    token_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(args.device)
    outputs = model.generate(token_ids, return_dict_in_generate=True, max_new_tokens=1, output_scores=True,
                             pad_token_id=PAD_TOKEN_ID, return_tensors="pt")

    output_logits = outputs.scores[0][0]
    zero_one_logits = output_logits[[ZERO_TOKEN_ID, ONE_TOKEN_ID]]
    prob = torch.softmax(zero_one_logits, dim=-1)[1].item()

    return prob

def pom_to_pgm(pom_model: PomBN):
    """
    Convert a pomegranate BayesianNetwork to a pgmpy BayesianNetwork.
    """
    # turn the pomegranate model into a dict
    model_dict = pom_model.to_dict()

    cpds, node_names, edges = [], [], []
    # compute the cpds
    for i, state in enumerate(model_dict['states']):
        node_names.append(state['name'])

        state_dist = state["distribution"]

        # If the node depends on parents, we need to handle its cpd differently
        if "parents" in state_dist:
            parent_names = []
            for parent_idx in model_dict["structure"][i]:
                parent_name = model_dict["states"][parent_idx]["name"]
                parent_names.append(parent_name)
                edges.append((parent_name, state["name"]))
            # turn the probability table into a pgmpy cpd
            table = state_dist["table"]
            probs = np.array([x[-1] for x in table])
            probs_pivoted = np.flip(probs.reshape(2, probs.shape[0] // 2, order='F'))
            cpds.append(TabularCPD(
                state["name"],
                2,
                probs_pivoted,
                evidence=parent_names,
                evidence_card=[2] * len(parent_names)
            ))
        else:
            # turn the state distribution into a cpd
            probs = [[state_dist["parameters"][0]["False"]],
                     [state_dist["parameters"][0]["True"]]]
            cpds.append(TabularCPD(
                state["name"],
                2,
                probs
            ))

    # create the pgmpy model
    pgm_net = PgmBN()
    pgm_net.add_nodes_from(node_names)
    pgm_net.add_edges_from(edges)
    pgm_net.add_cpds(*cpds)
    return pgm_net
