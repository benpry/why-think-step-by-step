#!/bin/bash

# parameters used in making Bayes nets
export N_NETS=100
export N_NODES=100
export N_EDGES=100

# Parameters used in making training data
export N_TRAIN=1000000
export NET_ID=33
export NUM_PAIRS=25
export EXP_P=0.5
export ZIPF_K=2

# Training / checkpointing stuff
export TOTAL_CHECKPOINTS=1000
export CHECKPOINT_INTERVAL=30000
export N_EPOCHS=50  # too many epochs
export N_TRAIN_STEPS=300001  # but we limit the number of training steps
export NUM_SAMPLES=10

export condition="local_joint_exp"
export SAMPLE_FORMAT=$condition
export SAMPLE_FORMAT_STR=$condition
CONDITION_WITH_HYPHEN=$(echo $condition | tr _ -)
export MODEL_NAME="${CONDITION_WITH_HYPHEN}-net-${NET_ID}"

base_archs=( "/alternate-base-lms/tiny" "/alternate-base-lms/small" "/alternate-base-lms/different" )

for base_arch in "${base_archs[@]}"; do
    export BASE_MODEL_PATH=$DATA_ROOT_DIR$base_arch
    export BASE_MODEL_NAME="${BASE_MODEL_PATH##*/}"
    export MODEL_ROOT_FOLDER="$DATA_ROOT_DIR/$BASE_MODEL_NAME"
    echo "Base model path: $BASE_MODEL_PATH"
    echo "Base model name: $BASE_MODEL_NAME"
    echo "Model root folder: $MODEL_ROOT_FOLDER"
    make results
    make eval_results
done
