#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
daytimes=( "day" "night")


for daytime in "${daytimes[@]}"
do

  echo "daytime: $daytime"
  eval_files="./src/splits/g2d/real_test_${daytime}.txt"
  python src/eval.py \
         --data_dir data \
         --min_depth 0.1 \
         --max_depth 100.0 \
         --height  512 \
         --width   1024 \
         --load_weights_folder models/g2d \
         --results_dir results/g2d \
         --eval_files_path $eval_files \
         --dataset g2d \
         --g2d_crop \
         --gen_figs

done