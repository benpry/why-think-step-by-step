"""
Define the architecture and tokenizer to train from
"""
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast
from pyprojroot import here
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--save-path", type=str)

if __name__ == "__main__":

    args = parser.parse_args()

    # set up the model
    config = GPT2Config(
        vocab_size=1024,
        n_embd=512,
        n_layer=10,
        n_head=8,
        bos_token_id=1023,
        eos_token_idx=1023
    )
    model = GPT2Model(config)
    model.save_pretrained(args.save_path)

    tokenizer = GPT2TokenizerFast(
            vocab_file=here("data/tokenizer/vocab.json"),
            merges_file=here("data/tokenizer/merges.txt")
    )
    tokenizer.save_pretrained(args.save_path)

