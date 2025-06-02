# MLOps Study

| version 1.0

## Flow

![flow](https://raw.githubusercontent.com/minyeamer/mlops-study/refs/heads/main/.images/flow.svg)

## Structure

```bash
./
├── data/
│   ├── test.csv # 테스트 데이터
│   └── train.csv # 학습 데이터
├── env/ # 환경 변수
├── models/ # 모델 파라미터
├── requirements.txt # 의존성
├── setup.sh # 가상환경 스크립트
└── src/
    ├── __init__.py
    ├── model.py # 모델 불러오기
    ├── predict.py # 예측
    ├── preprocessing.py # 데이터 전처리
    ├── train.py # 학습 및 평가
    └── utils.py # 공통 로직 보관
```