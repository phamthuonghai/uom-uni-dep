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
LANG=en-ud

mkdir ./saves/phanxu/${LANG}
${PYTHON} -u ./phanxu/phanxu.py all -m ./saves/phanxu/${LANG}/ -i ./saves/udpipe/${LANG}-train.conllu -i ./saves/bmst/${LANG}/train.conllu -g ./data/treebanks/${LANG}-train.conllu -t ./saves/udpipe/${LANG}-dev.conllu -t ./saves/bmst/${LANG}/dev.conllu -o ./saves/phanxu/${LANG}/dev.conllu -e ./data/fasttext/wiki.en.vec
date