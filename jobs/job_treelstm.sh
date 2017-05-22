#!/usr/bin/env bash
#
# Select Q
#$ -q albert.q
#
# Your job name
#$ -N treelstm
#
# Use current working directory
#$ -cwd
#
# Join stdout and stderr
#$ -j y
#$ -o ./jobs/log_treelstm.out
#
# Run job through bash shell
#$ -S /bin/bash
#
PYTHON=$(pwd)/venv-gpu/bin/python
source $(pwd)/venv-gpu/bin/activate
LANG=grc-ud

mkdir ./saves/treelstm/${LANG}

# bmst
# python bmstparser/parser.py --test saves/gold/${LANG}/train.conllu --outdir saves/bmst/${LANG}/ --params saves/bmst/${LANG}/en.params.pickle --model saves/bmst/${LANG}/en.model --predict
# mv saves/bmst/${LANG}/test_pred.conllu saves/bmst/${LANG}/train.conllu

# udpipe
# mkdir -p saves/udpipe/${LANG}
# ./udpipe/src/udpipe --parse saves/udpipe/en-ud/model.udpipe saves/gold/en-ud/train.conllu > saves/udpipe/en-ud/train.conllu

date
${PYTHON} -u ./phanxu/train_treelstm.py --tree_1_dir saves/udpipe/en-ud/ --tree_2_dir saves/bmst/en-ud/ --tree_gold_dir saves/gold/en-ud/ --embedding_file data/fasttext/en.vec --checkpoint_base saves/treelstm/en-ud/
date
