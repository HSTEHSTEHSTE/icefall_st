#!/usr/bin/env bash

source /home/hltcoe/xli/.bashrc
source /home/hltcoe/xli/anaconda3/etc/profile.d/conda.sh
CUDA_VISIBLE_DEVICES=$(free-gpu)

conda activate k2
cd /home/hltcoe/xli/ssm/icefall_st/egs/librispeech/ASR/
python conformer_ctc3/train.py