#!/bin/bash

source setup.sh

# 함수 정의: 모델 학습
train_model() {
    source scripts/data_train_options.sh
    source scripts/model_options.sh
    source scripts/train_options.sh

    echo "=== 모델 학습 시작 ==="
    python -m src.train --data "$data_options" --model "$model_options" --train "$train_options"
    if [ $? -ne 0 ]; then
        echo "오류: 모델 학습 중 문제가 발생했습니다. 작업을 종료합니다."
        exit 1
    fi
    echo "=== 모델 학습 완료 ==="
}

# 함수 정의: 예측 실행
run_prediction() {
    source scripts/data_test_options.sh
    source scripts/model_options.sh
    save_to="{DATA_DIR}/predictions.csv"

    echo "=== 예측 시작 ==="
    read -p "저장 위치 ($save_to): " user_input
    if [ -n "$user_input" ]; then
        save_to="$user_input"
    fi
    python -m src.predict --data "$data_options" --model "$model_options" --save_to "$save_to"
    echo "=== 예측 종료 ==="
}

ask_continue() {
    read -p "계속하려면 아무 키나 누르십시오..." user_input
}

while true; do
    clear
    echo "다음 중 실행할 작업을 선택하세요:"
    echo "1) 모델 학습"
    echo "2) 예측 실행"
    echo "3) 모델 학습 후 예측 실행"
    echo "4) 종료"
    read -p "선택 (1-4): " choice
    case $choice in
        1)
            train_model && ask_continue
            ;;
        2)
            if [ -f "models/logistic_model.joblib" ] || [ -f "models/xb_model.json" ]; then
                run_prediction
            else
                echo "오류: 학습된 모델이 없습니다. 먼저 모델을 학습해주세요."
            fi
            ask_continue
            ;;
        3)
            train_model && run_prediction && ask_continue
            ;;
        4)
            echo "작업을 종료합니다."
            exit 0
            ;;
        *)
            echo "잘못된 선택입니다. 1-4 중에서 선택해주세요."
            ;;
    esac
done 

