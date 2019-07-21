#!/bin/bash

WORKDIR="BERT_01"
if [ ! -d "$WORKDIR" ]; then
    mkdir $WORKDIR
    ln -s ../data $WORKDIR/data
    cp original_label_hierarchy.txt $WORKDIR/original_label_hierarchy.txt
fi
cd $WORKDIR

SAMPLING_MINMAX_RATIO=3
LOSS_MINMAX_RATIO=1
NUM_EPOCHS=10
LOSS="bce"
echo "SAMPLING_MINMAX_RATIO = ${SAMPLING_MINMAX_RATIO}" > params.txt
echo "LOSS_MINMAX_RATIO = ${LOSS_MINMAX_RATIO}" >> params.txt
echo "NUM_EPOCHS = ${NUM_EPOCHS}" >> params.txt
echo "LOSS = ${LOSS}" >> params.txt
PYTHONPATH=../ python ../subtask_a_pretrained_bert.py
cd..


####
WORKDIR="BERT_02"
if [ ! -d "$WORKDIR" ]; then
    mkdir $WORKDIR
    ln -s ../data $WORKDIR/data
    cp original_label_hierarchy.txt $WORKDIR/original_label_hierarchy.txt
fi
cd $WORKDIR

SAMPLING_MINMAX_RATIO=3
LOSS_MINMAX_RATIO=3
NUM_EPOCHS=10
LOSS="bce"
echo "SAMPLING_MINMAX_RATIO = ${SAMPLING_MINMAX_RATIO}" > params.txt
echo "LOSS_MINMAX_RATIO = ${LOSS_MINMAX_RATIO}" >> params.txt
echo "NUM_EPOCHS = ${NUM_EPOCHS}" >> params.txt
echo "LOSS = ${LOSS}" >> params.txt
PYTHONPATH=../ python ../subtask_a_pretrained_bert.py
cd ..

###
WORKDIR="BERT_03"
if [ ! -d "$WORKDIR" ]; then
    mkdir $WORKDIR
    ln -s ../data $WORKDIR/data
    cp original_label_hierarchy.txt $WORKDIR/original_label_hierarchy.txt
fi
cd $WORKDIR
SAMPLING_MINMAX_RATIO=3
LOSS_MINMAX_RATIO=6
NUM_EPOCHS=10
LOSS="bce"
echo "SAMPLING_MINMAX_RATIO = ${SAMPLING_MINMAX_RATIO}" > params.txt
echo "LOSS_MINMAX_RATIO = ${LOSS_MINMAX_RATIO}" >> params.txt
echo "NUM_EPOCHS = ${NUM_EPOCHS}" >> params.txt
echo "LOSS = ${LOSS}" >> params.txt
PYTHONPATH=../ python ../subtask_a_pretrained_bert.py
cd ..


###
WORKDIR="BERT_04"
if [ ! -d "$WORKDIR" ]; then
    mkdir $WORKDIR
    ln -s ../data $WORKDIR/data
    cp original_label_hierarchy.txt $WORKDIR/original_label_hierarchy.txt
fi
cd $WORKDIR
SAMPLING_MINMAX_RATIO=6
LOSS_MINMAX_RATIO=3
NUM_EPOCHS=10
LOSS="bce"
echo "SAMPLING_MINMAX_RATIO = ${SAMPLING_MINMAX_RATIO}" > params.txt
echo "LOSS_MINMAX_RATIO = ${LOSS_MINMAX_RATIO}" >> params.txt
echo "NUM_EPOCHS = ${NUM_EPOCHS}" >> params.txt
echo "LOSS = ${LOSS}" >> params.txt
PYTHONPATH=../ python ../subtask_a_pretrained_bert.py
cd ..