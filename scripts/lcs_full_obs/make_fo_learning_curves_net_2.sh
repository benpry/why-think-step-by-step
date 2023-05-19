#!/bin/bash

# parameters used in making Bayes nets
export N_NETS=100
export N_NODES=100
export N_EDGES=100

# Parameters used in making training data
export N_TRAIN=1000000
export NET_ID=2
export NUM_PAIRS=25
export EXP_P=0.5
export ZIPF_K=2

# Training / checkpointing stuff
export TOTAL_CHECKPOINTS=1000
export CHECKPOINT_INTERVAL=30000
export N_EPOCHS=50  # too many epochs
export N_TRAIN_STEPS=300001  # but we limit the number of training steps

# Fine-tuning and beyond
export MODEL_ROOT_FOLDER=$DATA_ROOT_DIR/300k-models
export BASE_MODEL_PATH=$MODEL_ROOT_FOLDER/base-lm

conditions=( "fully_observed" )

for condition in "${conditions[@]}"; do
    export SAMPLE_FORMAT=$condition
    export SAMPLE_FORMAT_STR=$condition
    CONDITION_WITH_HYPHEN=$(echo $condition | tr _ -)
    export MODEL_NAME="${CONDITION_WITH_HYPHEN}-net-${NET_ID}"
    make learning_curves
done
