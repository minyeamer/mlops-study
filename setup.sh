#!/bin/bash

venv_path=".venv"

if [ -z "$VIRTUAL_ENV" ]; then
    read -p "가상환경 경로를 입력해주세요: " -p user_input 
    if [ -n "$user_input" ]; then
        venv_path="$user_input"
    fi
    if [ -d "$venv_path" ]; then
        source "${venv_path}/bin/activate"
    else
        read -p "가상환경이 존재하지 않습니다. 가상환경을 생성하겠습니까? (y/n) " user_input
        if [[ "$user_input" ==  "y" || "$user_input" ==  "Y" ]]; then
            python -m venv "$venv_path"
            source "${venv_path}/bin/activate"
            if [ -z "$VIRTUAL_ENV" ]; then
                echo "오류: 가상환경이 생성 중 문제가 발생했습니다. 작업을 종료합니다."
                exit 0
            else
                if [ -d "requirements.txt" ]; then
                    echo "True"
                    # pip install -r requirements.txt
                fi
            fi
        else
            echo "작업을 종료합니다."
        fi
    fi
else
    echo "가상환경이 확인되었습니다: ${VIRTUAL_ENV}"
fi