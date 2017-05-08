#!/usr/bin/env bash
#
# Select Q
#$ -q main.q
#
# Your job name
#$ -N bmst
#
# Use current working directory
#$ -cwd
#
# Join stdout and stderr
#$ -j y
#$ -o ./jobs/log_bmst.out
#
# Run job through bash shell
#$ -S /bin/bash
#

PYTHON=$(pwd)/venv/bin/python
source ./venv/bin/activate



files="./data/treebanks/*-ud-dev.conllu"
regex="([a-z_]+)-ud-dev.conllu"

for f in $files
do
    if [[ $f =~ $regex ]]
    then
        echo "=========================================================================="
        echo "                          LANGUAGE: ${BASH_REMATCH[1]}"
        echo "=========================================================================="
        TRAIN_FILE="./data/treebanks/${BASH_REMATCH[1]}-ud-train.conllu"
        DEV_FILE="./data/treebanks/${BASH_REMATCH[1]}-ud-dev.conllu"
        MODEL_DIR="./saves/bmst/${BASH_REMATCH[1]}-ud"
        mkdir -p ${MODEL_DIR}
        ${PYTHON} -u ./bmstparser/parser.py --outdir ${MODEL_DIR} --train ${TRAIN_FILE} --dev ${DEV_FILE} --epochs 30 --lstmdims 125 --lstmlayers 2 --bibi-lstm
    fi
done
