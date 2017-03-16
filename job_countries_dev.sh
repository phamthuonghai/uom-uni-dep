#!/bin/sh
#
# Select Q
#$ -q albert.q
#
# Your job name
#$ -N chen_parser
#
# Use current working directory
#$ -cwd
#
# Join stdout and stderr
#$ -j y
#$ -o countries-dev.out
#
# Run job through bash shell
#$ -S /bin/bash
#
export PYTHONPATH=$PYTHONPATH:$(pwd)
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=0.4
PYTHON=$(pwd)/venv/bin/python

TEMPLATE_FILE=./config/chen.template

files="./data/ud-treebanks-conll2017/*/*-ud-dev.conllu"
regex="UD_([A-Za-z\-]+)\/([a-z_]+)-ud-dev.conllu"

for f in $files
do
    if [[ $f =~ $regex ]]
    then
        echo "=========================================================================="
        echo "                          LANGUAGE: ${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
        echo "=========================================================================="

        TRAIN_INPUT="./data/ud-treebanks-conll2017/UD_${BASH_REMATCH[1]}/${BASH_REMATCH[2]}-ud-train.conllu"
        FEATURE_FILE="./models/${BASH_REMATCH[2]}-ud-train-ft.pkl"
        MODEL_PREFIX="./models/${BASH_REMATCH[2]}-ud-train"
        PARSE_INPUT="./data/ud-treebanks-conll2017/UD_${BASH_REMATCH[1]}/${BASH_REMATCH[2]}-ud-dev.conllu"
        PARSE_OUTPUT="./models/${BASH_REMATCH[2]}-ud-res.conllu"

        echo "========================= FEATURES EXTRACTION ${BASH_REMATCH[2]} ========================="
        $PYTHON -m chen_parser.feature -t $TEMPLATE_FILE $TRAIN_INPUT $FEATURE_FILE
        echo "=========================== TRAINING ORACLE ${BASH_REMATCH[2]} ==========================="
        $PYTHON -m chen_parser.oracle $FEATURE_FILE $MODEL_PREFIX
        echo "========================== PARSING TEST DATA ${BASH_REMATCH[2]} =========================="
        $PYTHON -m chen_parser.parser -t $TEMPLATE_FILE $PARSE_INPUT $MODEL_PREFIX $PARSE_OUTPUT
        echo "=========================== TESTING RESULTS ${BASH_REMATCH[2]} ==========================="
        $PYTHON ./conll/evaluation_script/conll17_ud_eval.py --verbose $PARSE_INPUT $PARSE_OUTPUT
    fi
done
