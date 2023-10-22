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
export NUM_SAMPLES=1

# Fine-tuning and beyond
export MODEL_ROOT_FOLDER=$DATA_ROOT_DIR/learning-curve-models
export BASE_MODEL_PATH=$MODEL_ROOT_FOLDER/base-lm
export BASE_MODEL_NAME="${BASE_MODEL_PATH##*/}"

conditions=( "local_joint_exp" "local_joint_zipf" "fully_observed_held_out" "wrong_local_joint_exp" "wrong_local_joint_zipf" )
nums_samples=( 1 2 3 4 5 6 7 8 9 10 )

for condition in "${conditions[@]}"; do

    if [ "$condition" == "local_joint_exp" ] || [ "$condition" == "fully_observed_held_out" ];
    then
        export MODEL_ROOT_FOLDER=$DATA_ROOT_DIR/learning-curve-models
    else
        export MODEL_ROOT_FOLDER=$DATA_ROOT_DIR
    fi
    export SAMPLE_FORMAT=$condition
    export SAMPLE_FORMAT_STR=$condition
    CONDITION_WITH_HYPHEN=$(echo $condition | tr _ -)
    export MODEL_NAME="${CONDITION_WITH_HYPHEN}-net-${NET_ID}"

    for num_samples in "${nums_samples[@]}"; do
        export NUM_SAMPLES=$num_samples
        make data/evaluation/base-model-${BASE_MODEL_NAME}/free-gen-probabilities-${MODEL_NAME}-${NUM_SAMPLES}samples.csv
        make data/evaluation/base-model-${BASE_MODEL_NAME}/scaffolded-gen-probabilities-${MODEL_NAME}-${NUM_SAMPLES}samples.csv
        make data/evaluation/base-model-${BASE_MODEL_NAME}/negative-scaffolded-gen-probabilities-${MODEL_NAME}-${NUM_SAMPLES}samples.csv
    done

done
