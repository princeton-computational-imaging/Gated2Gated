#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
python src/inference.py \
       --data_dir               ./example \
       --height                 512 \
       --width                  1024 \
       --min_depth              0.1 \
       --max_depth              100.0 \
       --depth_normalizer       70.0 \
       --results_dir            ./results \
       --weights_dir            ./models/stf