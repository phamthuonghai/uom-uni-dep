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
#$ -o chen.out
#
# Run job through bash shell
#$ -S /bin/bash
#
export PYTHONPATH=$PYTHONPATH:$(pwd)
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=0.4
PYTHON=$(pwd)/venv/bin/python

TRAIN_INPUT=./data/es-ca-ud-train.conllu
FEATURE_FILE=./models/es-ca-ud-train-ft.pkl
MODEL_PREFIX=./models/es-ca-train
PARSE_INPUT=./data/ud-treebanks-conll2017/UD_Spanish/es-ud-dev.conllu
PARSE_OUTPUT=./models/es-ud-res.conllu

echo "========================== FEATURES EXTRACTION =========================="
cat $TRAIN_INPUT | $PYTHON -m chen_parser.feature -o $FEATURE_FILE -p 14187
echo "============================ TRAINING ORACLE ============================"
$PYTHON -m chen_parser.oracle $FEATURE_FILE $MODEL_PREFIX
echo "=========================== PARSING TEST DATA ==========================="
$PYTHON -m chen_parser.parser $PARSE_INPUT $MODEL_PREFIX $PARSE_OUTPUT
echo "============================ TESTING RESULTS ============================"
$PYTHON ./conll/evaluation_script/conll17_ud_eval.py --verbose $PARSE_INPUT $PARSE_OUTPUT
