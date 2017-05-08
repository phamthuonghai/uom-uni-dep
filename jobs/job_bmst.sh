#!/usr/bin/env bash
#
# Select Q
#$ -q albert.q
#
# Your job name
#$ -N bmst
#
# Use current working directory
#$ -cwd
#
# Join stdout and stderr
#$ -j y
#$ -o log_bmst.out
#
# Run job through bash shell
#$ -S /bin/bash
#

TRAIN_FILE=./data/treebanks/grc-ud-train.conllu
DEV_FILE=./data/treebanks/grc-ud-dev.conllu
MODEL_DIR=./saves/grc-ud
source ./venv/bin/activate
./venv/bin/python -u ./bmstparser/parser.py --outdir ${MODEL_DIR} --train ${TRAIN_FILE} --dev ${DEV_FILE} --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm