#!/bin/bash

PARTITION="hard"
NODELIST=("thin" "pascal")
# NODELIST=("modjo" "daft" "kavinsky" "punk" )
# NODELIST=("kavinsky" "daft" "modjo")
NUM_NODES=${#NODELIST[@]}

DATASET_NAME=$1
LANG=$2
DRY_RUN=$3

OUTPUT_DIR=./slurm_output

PROMPT_TYPES=("standard" "AT-standard" "CoT" "AT-CoT")

for pt in "${PROMPT_TYPES[@]}"; do
  JOB_NAME=${pt}
  sbatch --nodes=1 --partition=$PARTITION --nodelist=$NODE --gres=gpu:1 --time=6-00:00:00 --gpu_partition ${PARTITION} --gpu_node $NODE \
         --job-name $JOB_NAME --output $OUTPUT_DIR/$JOB_NAME.out --error $OUTPUT_DIR/$JOB_NAME.err \
         run_python.sh --dataset_name $DATASET_NAME --prompt_type ${pt} --lang ${LANG} --dry_run $DRY_RUN 
done

