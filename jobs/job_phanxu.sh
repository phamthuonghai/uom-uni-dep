#!/usr/bin/env bash
#
# Select Q
#$ -q albert.q
#
# Your job name
#$ -N phanxu
#
# Use current working directory
#$ -cwd
#
# Join stdout and stderr
#$ -j y
#$ -o ./jobs/log_phanxu.out
#
# Run job through bash shell
#$ -S /bin/bash
#
date
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=0.4,dnn.enabled=False

PYTHON=$(pwd)/venv/bin/python
source $(pwd)/venv/bin/activate

${PYTHON} -u ./phanxu/phanxu.py train -i ./saves/udpipe/grc-ud-train.conllu -i ./saves/grc-ud/train.conllu -g ./data/treebanks/grc-ud-train.conllu
date