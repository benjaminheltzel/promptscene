#!/bin/bash

#cd ../..

# custom config
DATA=$3
TRAINER=MaPLePromptScene

DATASET=$1
SEED=$2

CFG=$4
SHOTS=-1


DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/replica/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --eval-only \
    # --load-epoch 200 
fi