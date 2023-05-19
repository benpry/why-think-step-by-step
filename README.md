# Code and data for "Why think step by step? Reasoning emerges from the locality of experience"

This directory contains code and most of the data for "Why think step by step? Reasoning emerges from the locality of experience.". 

## Code

Code is stored in the `code` directory.

The `visualize` subdirectory in the code directory contains the code used to make plots and tables. They are all Quarto Markdown files, but contain entirely R code.

The `core` subdirectory contains useful utilities for common objects and tasks, including the representation of Bayes nets and utilities for getting probabilities out of transformers.

The `evaluate` subdirectory contains files that compute estimated or true probabilities, as well as mutual information between Bayes nets.

The `finetune` subdirectory contains the script which trains a language model on a given dataset.

The `make_train_data` subdirectory contains code that randomly generates the Bayes nets, selects pairs to hold out from the training set, and generates the training set.

The `scaffold` subdirectory contains code that computes both true scaffolds and negative scaffolds for scaffolded generation.

The `train` subdirectory contains code that trains the tokenizer and defines the base language model with random weights that all other models are trained from.

### Condition names

The names of conditions in the code are not always the same as how they are described in the paper, so we describe the naming in the code here.

`fully-observed-held-out` is the fully-observed condition reported in the paper.
`fully-observed` is fully-observed with no held-out pairs and is used only in the data efficiency section.
`local-joint-exp` is local (geom). 
`local-joint-zipf` is local (Zipf).
`wrong-local-joint-exp` is wrong local (geom).
`wrong-local-joint-zipf` is wrong local (zipf).

The names of estimators are self-explanatory, except for that `fixed` refers to direct prediction.

## scripts

We use `make` to run most of our data generation, training, and analysis pipelines. Each script sets the parameters that get used in the Makefile and calls `make` with the appropriate target. Each script corresponds to a single Bayes net.

## data

The `evaluation` subdirectory contains true conditional and marginal probabilities (in `true-probs`), data for generating learning curves (in `learning-curves`), mutual informations between variable pairs in each Bayes net (in `mutual-informations`), and estimated probabilities for different combinations of training condition and estimator.

The `scaffolds` subdirectory includes both scaffolds and negative scaffolds for each selected Bayes net.

The `tokenizer` subdirectory contains the trained tokenizer.

The `training-data` subdirectory contains the training data. Currently, it just contains the selected variable pairs for each Bayes net, along with the mean mutual informations for each set of selected pairs. The `samples` subdirectory is empty, but it would have stored the samples that we used as training data for the language models. We could not include all the training data because of the size limit, but training data can be generated using the code.

There is also no `bayes_nets` subdirectory in `data` because of capacity limits, but Bayes nets can also be generated using our code.
