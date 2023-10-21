"""#!/usr/bin/env python
Define base language models of different sizes to test robustness to model type.
"""
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast
from pyprojroot import here
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--base-path", type=str)

if __name__ == "__main__":

    args = parser.parse_args()

    small_config = GPT2Config(
        vocab_size=1024,
        n_embd=256,
        n_layer=5,
        n_head=4,
        bos_token_id=1023,
        eos_token_idx=1023
    )
    different_config = GPT2Config(
        vocab_size=1024,
        n_embd=1024,
        n_layer=3,
        n_head=8,
        bos_token_id=1023,
        eos_token_idx=1023
    )
    large_config = GPT2Config(
        vocab_size=1024,
        bos_token_id=1023,
        eos_token_idx=1023
    )
    tiny_config = GPT2Config(
        vocab_size=1024,
        n_embd=4,
        n_layer=2,
        n_head=2,
        bos_token_id=1023,
        eos_token_idx=1023
    )

    small_model = GPT2Model(small_config)
    small_model.save_pretrained(args.base_path + "/small")

    different_model = GPT2Model(different_config)
    different_model.save_pretrained(args.base_path + "/different")

    large_model = GPT2Model(large_config)
    large_model.save_pretrained(args.base_path + "/large")

    tiny_model = GPT2Model(tiny_config)
    tiny_model.save_pretrained(args.base_path + "/tiny")

    # save the same tokenizer for all models
    tokenizer = GPT2TokenizerFast(
            vocab_file=here("data/tokenizer/vocab.json"),
            merges_file=here("data/tokenizer/merges.txt")
    )
    tokenizer.save_pretrained(args.base_path + "/small")
    tokenizer.save_pretrained(args.base_path + "/different")
    tokenizer.save_pretrained(args.base_path + "/large")
    tokenizer.save_pretrained(args.base_path + "/tiny")
