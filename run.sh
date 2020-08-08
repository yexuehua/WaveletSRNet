#!/usr/bin/env sh
#
#$ -cwd
#$ -j y
#$ -N output_train_lstm
#$ -S /bin/sh
#
python main.py --ngpu=1 --test --start_epoch=0 --test_iter=1000 --batchSize=64 --test_batchSize=32 --nrow=4 --upscale=2 --lr=2e-4
