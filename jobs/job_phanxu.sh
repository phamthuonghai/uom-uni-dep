#!/usr/bin/env bash
#
# Select Q
#$ -q main.q
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

for LANG in 'en-ud' 'grc-ud'
do
    echo "==UDPIPE=="
    ${PYTHON} -u conll/evaluation_script/conll17_ud_eval.py --verbose ./saves/gold/${LANG}/test.conllu ./saves/udpipe/${LANG}/test.conllu
    echo "==BSMT=="
    ${PYTHON} -u conll/evaluation_script/conll17_ud_eval.py --verbose ./saves/gold/${LANG}/test.conllu ./saves/bmst/${LANG}/test.conllu
    for sent_type in 'sum' 'lstm'
    do
        for model_type in 'compare' 'score'
        do
            date
            mkdir ./saves/phanxu/${LANG}/${sent_type}-${model_type}
            echo "==========${sent_type} ${model_type}========"
            model_path=./saves/phanxu/${LANG}/${sent_type}-${model_type}/
            ${PYTHON} -u ./phanxu/phanxu.py all -m ${model_path} -i ./saves/udpipe/${LANG}/train.conllu -i ./saves/bmst/${LANG}/train.conllu -g ./saves/gold/${LANG}/train.conllu -t ./saves/udpipe/${LANG}/test.conllu -t ./saves/bmst/${LANG}/test.conllu -o ${model_path}/test.conllu -e ./data/fasttext/${LANG}.vec -c ${model_type} -s ${sent_type} > ./jobs/log_phanxu_${sent_type}_${model_type}.out 2>&1
            ${PYTHON} -u conll/evaluation_script/conll17_ud_eval.py --verbose ./saves/gold/${LANG}/test.conllu ${model_path}/test.conllu
            date
        done
    done
done