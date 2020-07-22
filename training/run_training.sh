#!/bin/bash
set -e
cd detector
eps=100
#CUDA_VISIBLE_DEVICES=1,2,3 python main.py --model res18 -b 12 --epochs $eps --save-dir res18_extend --gpu 1,2,3 --save-freq 1
CUDA_VISIBLE_DEVICES=1,2,3 python main.py --model res18 -b 3 --resume results/res18_extend/004.ckpt --test 1 --gpu 1,2,3
#--resume /data/wzeng/DSB_3/training/detector/results/res18/005.ckpt