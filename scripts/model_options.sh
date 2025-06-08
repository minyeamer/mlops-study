#!/bin/bash

model_type="best"
model_path=""
model_params=""

model_selection=$(cat <<EOF
모델 종류 또는 모델이 저장된 경로를 입력하시겠습니까? (기본값 안내)
  모델 종류: $model_type
[y/n] 
EOF
)

read -p "${model_selection}" set_data_yn
if [[ "$set_data_yn" ==  "y" || "$set_data_yn" ==  "Y" ]]; then
    echo "옵션을 입력해주세요. (아무것도 입력하지 않으면 괄호 안 기본값이 적용됩니다)"
    read -p "모델 종류 ($model_type): " user_input
    if [ -n "$user_input" ]; then
        model_type="$user_input"
    fi
    read -p "모델 경로 ($model_path): " user_input
    if [ -n "$user_input" ]; then
        model_path="$user_input"
    fi
    read -p "모델 파라미터, URL 형식 ($model_params): " user_input
    if [ -n "$user_input" ]; then
        model_params="$user_input"
    fi
fi

model_options=$(cat <<EOF | tr -d '\n'
model_type=$model_type
&model_path=$model_path
&model_params=$model_params
EOF
)