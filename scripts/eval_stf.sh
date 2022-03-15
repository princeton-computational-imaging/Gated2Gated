#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
weathers=( "clear" "light_fog" "dense_fog" "snow" )
daytimes=( "day" "night")


for daytime in "${daytimes[@]}"
do
  for weather in "${weathers[@]}"
  do
    echo "daytime: $daytime, weather: $weather"
    eval_files="./src/splits/stf/test_${weather}_${daytime}.txt"
    python src/eval.py \
           --data_dir data \
           --min_depth 0.1 \
           --max_depth 100.0 \
           --height  512 \
           --width   1024 \
           --load_weights_folder models/stf \
           --results_dir results/stf \
           --eval_files_path $eval_files \
           --dataset stf \
           --g2d_crop \
           --gen_figs \
           --binned_metrics
  done
done