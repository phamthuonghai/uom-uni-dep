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
#$ -o chen-nolabel.out
#
# Run job through bash shell
#$ -S /bin/bash
#
export PYTHONPATH=$PYTHONPATH:$(pwd)
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
export PYTHON=$(pwd)/venv/bin/python

export TRAIN_INPUT=./data/ud-treebanks-conll2017/UD_English/en-ud-train.conllu
export FEATURE_FILE=./models/en-ud-train-nolabel-ft.pkl
export MODEL_PREFIX=./models/en-ud-train-nolabel
export PARSE_INPUT=./data/ud-treebanks-conll2017/UD_English/en-ud-dev.conllu
export PARSE_OUTPUT=./models/en-ud-res-nolabel.conllu
export TEMPLATE_FILE=./config/chen-nolabel.template

echo "========================== FEATURES EXTRACTION =========================="
$PYTHON -m chen_parser.feature -t $TEMPLATE_FILE $TRAIN_INPUT $FEATURE_FILE
echo "============================ TRAINING ORACLE ============================"
$PYTHON -m chen_parser.oracle $FEATURE_FILE $MODEL_PREFIX
echo "=========================== PARSING TEST DATA ==========================="
$PYTHON -m chen_parser.parser -t $TEMPLATE_FILE $PARSE_INPUT $MODEL_PREFIX $PARSE_OUTPUT
echo "============================ TESTING RESULTS ============================"
$PYTHON ./conll/evaluation_script/conll17_ud_eval.py --verbose $PARSE_INPUT $PARSE_OUTPUT
