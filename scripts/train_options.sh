#!/bin/bash

epoch=""
batch_size=""
shuffle="true"
save_mode="best_only"

train_selection=$(cat <<EOF
모델 학습 조건을 입력하시겠습니까? (기본값 안내)
  Epoch: $epoch
  Batch Size: $batch_size
  Shuffle: $shuffle
  저장 방식: $save_mode
[y/n] 
EOF
)

read -p "${train_selection}" set_data_yn
if [[ "$set_data_yn" ==  "y" || "$set_data_yn" ==  "Y" ]]; then
    echo "옵션을 입력해주세요. (아무것도 입력하지 않으면 괄호 안 기본값이 적용됩니다)"
    read -p "Epoch ($epoch): " user_input
    if [ -n "$user_input" ]; then
        epoch="$user_input"
    fi
    read -p "Batch Size ($batch_size): " user_input
    if [ -n "$user_input" ]; then
        batch_size="$user_input"
    fi
    read -p "Shuffle ($shuffle): " user_input
    if [ -n "$user_input" ]; then
        shuffle="$user_input"
    fi
    read -p "저장 방식 ($save_mode): " user_input
    if [ -n "$user_input" ]; then
        save_mode="$user_input"
    fi
fi

train_options=$(cat <<EOF | tr -d '\n'
epoch=$epoch
&batch_size=$batch_size
&shuffle=$shuffle
EOF
)