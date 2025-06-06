#!/bin/bash

source scripts/data_options.sh
# echo "$data_options"

source scripts/model_options.sh
# echo "$model_options"

source scripts/train_options.sh
# echo "$train_options"

echo "=== 모델 학습 시작 ==="

python -m src.train --data "$data_options" --model "$model_options" --train "$train_options"

echo "=== 모델 학습 완료 ==="