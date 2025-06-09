#!/bin/bash

file_path="{DATA_DIR}/test.csv"
drop_columns="UID"
numeric_columns=""
categorical_columns=""

data_selection=$(cat <<EOF
예측 데이터 경로 및 전처리 옵션을 입력하시겠습니까? (기본값 안내)
  파일 경로: $file_path
  제거할 열: $drop_columns
[y/n] 
EOF
)

read -p "${data_selection}" set_data_yn
if [[ "$set_data_yn" ==  "y" || "$set_data_yn" ==  "Y" ]]; then
    echo "옵션을 입력해주세요. (아무것도 입력하지 않으면 괄호 안 기본값이 적용됩니다)"
    read -p "파일 경로 ($file_path): " user_input
    if [ -f "$user_input" ]; then
        file_path="$user_input"
    fi
    read -p "제거할 열, 콤마로 구분 ($drop_columns): " user_input
    if [ -n "$user_input" ]; then
        drop_columns="$user_input"
    fi
    read -p "숫자형 열, 콤마로 구분 ($numeric_columns): " user_input
    if [ -n "$user_input" ]; then
        numeric_columns="$user_input"
    fi
    read -p "범주형 열, 콤마로 구분 ($categorical_columns): " user_input
    if [ -n "$user_input" ]; then
        categorical_columns="$user_input"
    fi
fi

data_options=$(cat <<EOF | tr -d '\n'
file_path=$file_path
&drop_columns=$drop_columns
&numeric_columns=$numeric_columns
&categorical_columns=$categorical_columns
EOF
)