#!/bin/bash

# parameters used in making Bayes nets
export N_NETS=100
export N_NODES=100
export N_EDGES=100

# Parameters used in making training data
export N_TRAIN=1000000
export NUM_PAIRS=25
export EXP_P=0.5
export ZIPF_K=2
export NET_ID=28

# Training / checkpointing stuff
export TOTAL_CHECKPOINTS=20
export CHECKPOINT_INTERVAL=30000
export N_EPOCHS=50  # too many epochs
export N_TRAIN_STEPS=300001  # but we limit the number of training steps

# Fine-tuning and beyond
export MODEL_ROOT_FOLDER=$DATA_ROOT_DIR
export BASE_MODEL_PATH=$MODEL_ROOT_FOLDER/base-lm
export BASE_MODEL_NAME=base-lm
export NUM_SAMPLES=10

conditions=( "local_joint_zipf" "wrong_local_joint_exp" "wrong_local_joint_zipf" "fully_observed" )

for condition in "${conditions[@]}"; do
    export SAMPLE_FORMAT=$condition
    export SAMPLE_FORMAT_STR=$condition
    CONDITION_WITH_HYPHEN=$(echo $condition | tr _ -)
    export MODEL_NAME="${CONDITION_WITH_HYPHEN}-net-${NET_ID}"
    make results
done
