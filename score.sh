#!/bin/bash

PARTITION="funky"
NODELIST=("rodgers" "bernard" "pascal")
# NODELIST=("cal" "pas" "sister")
# NODELIST=("cal" "pas" "sister")
NUM_NODES=${#NODELIST[@]}

DATASET_NAME=$1
NOISE_TYPE=$2
SCORE_TYPE=$3
SCORE_STAGE=$4
TURN_ID=$5
DRY_RUN=$6

OUTPUT_DIR=./slurm_output

if [ "$SCORE_TYPE" == "cq" ]; then
  STAGE=generation
  NOISE_TYPE=1
  USER_SIMULATION_MODES=("respond")
else
  # USER_SIMULATION_MODES=("select" "respond")
  USER_SIMULATION_MODES=("select" "respond" "select+respond")
  # USER_SIMULATION_MODES=("select")
fi

PROMPT_TYPES=("few-shot" "AT-few-shot" "AT-CoT-few-shot")
# PROMPT_TYPES=("AT-CoT-few-shot")

job_counter=0

for usm in "${USER_SIMULATION_MODES[@]}"; do
  NODE=${NODELIST[$job_counter]} 
  for pt in "${PROMPT_TYPES[@]}"; do
    JOB_NAME=score-${DATASET_NAME}-${NOISE_TYPE}-${TURN_ID}-${usm}-${pt}
    sbatch --nodes=1 --partition=$PARTITION --nodelist=$NODE --gres=gpu:1 --time=6-00:00:00 \
            --job-name $JOB_NAME --output $OUTPUT_DIR/$JOB_NAME.out --error $OUTPUT_DIR/$JOB_NAME.err \
            run_score.sh --dataset_name $DATASET_NAME --turn_id $TURN_ID  --stage score --user_simulation_mode ${usm} --prompt_type ${pt} \
                         --noise_type $NOISE_TYPE --score_type $SCORE_TYPE --score_stage $SCORE_STAGE --dry_run $DRY_RUN --gpu_partition $PARTITION --gpu_node $NODE
  done
  job_counter=$((job_counter+1))
done

