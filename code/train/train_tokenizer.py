"""
Define a Byte Pair Encoding tokenizer, train on some data in our format, then save it.
"""
from tokenizers import ByteLevelBPETokenizer

if __name__ == "__main__":

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=["./data/training-data/samples/eval_samples_local_joint_exp_net_33.csv"], vocab_size=1024, min_frequency=2)
    tokenizer.save("./data/tokenizer/tokenizer.json")
