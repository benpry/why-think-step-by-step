"""
Count the number of parameters in each model.
"""
import os
from transformers import AutoModel

model_filepaths = [
    "/base-lm",
    "/alternate-base-lms/tiny",
    "/alternate-base-lms/small",
    "/alternate-base-lms/different",
    "/alternate-base-lms/large"
]

if __name__ == "__main__":

    root_dir = os.environ["DATA_ROOT_DIR"]

    for model_filepath in model_filepaths:

        model_name = model_filepath.split("/")[-1]
        print(f"model name: {model_name}")

        model = AutoModel.from_pretrained(root_dir + model_filepath)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total # parameters: {n_parameters}")
