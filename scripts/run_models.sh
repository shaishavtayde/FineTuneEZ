#!/bin/bash

#export CUDA_VISIBLE_DEVICES=2
#export CUDA_VISIBLE_DEVICES=""

LR_PATIENCE=2
LR=2e-5
#LR=2e-5
BATCH_SIZE=16
OPTIMIZER=adamw


#MODEL_NAME=hf-internal-testing/tiny-random-RobertaModel
MODEL_NAME=mrm8488/bert-tiny-finetuned-sms-spam-detection
INPUTDIR=/c/Users/SHAISHAV/Desktop/Projects/GPT-Instruction-Engine/dataset
OUTPUTDIR=/c/Users/SHAISHAV/Desktop/Projects/GPT-Instruction-Engine/models



OUT_DIR=$OUTPUTDIR
mkdir -p $OUT_DIR 
TRAIN_DATASET=${INPUTDIR}/train
VAL_DATASET=${INPUTDIR}/valid
TEST_DATASET=${INPUTDIR}/test
cmd="train-hf --model $MODEL_NAME --lr $LR --labeled-data $TRAIN_DATASET --validation-data $VAL_DATASET --batch-size $BATCH_SIZE --output-dir $OUT_DIR --optimizer $OPTIMIZER --save-initialization --lr-plateau-patience $LR_PATIENCE --num-labels 2 --max-training-epochs 1"
echo $cmd
eval $cmd


