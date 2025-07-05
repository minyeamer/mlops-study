## 프로젝트 개요

> Airflow와 MLflow를 효과적으로 사용하기 위한 모듈화를 진행

> <a href="https://github.com/minyeamer/mlops-study" target="_blank">https://github.com/minyeamer/mlops-study</a>

### 목표
- DACON의 <a href="https://dacon.io/competitions/official/236450/data" target="_blank">채무 불이행 여부 예측 해커톤: 불이행의 징후를 찾아라!</a> 경진대회에서 제공하는 데이터셋을 사용하여 배포/서빙을 위한 모듈화를 해보기
- 테스트를 목적으로 하며, 모듈은 10개 이하로 제한하고 필수 기능만 구현
- 성능보다는 프로젝트 구조와 파이썬 코드의 구성 방식에 중점을 둘 예정

### 흐름도

![flow](https://raw.githubusercontent.com/minyeamer/mlops-study/refs/heads/main/.images/flow.svg)

- 전처리(`preprocess.py`), 모델(`model.py`), 학습(`train.py`), 예측(`predict.py`) 4개의 모듈로 구성
- 학습/검증 데이터는 `data/` 경로 아래 CSV 파일을 활용
- `scripts/*_options.sh` 스크립트를 통해 파라미터를 입력받고 각 모듈 실행 시 전달
- `setup.sh` 스크립트를 통해 `requirements.txt` 에 명시된 의존성을 포함하는 가상환경을 활성화
- `run_service.sh` 스크립트에서 while 반복문을 돌면서 4개의 Mode 중 하나를 입력받아 작업 수행

### 모듈 & 스크립트 구조

```bash
./
├── data/
│   ├── train.csv # 학습 데이터
│   └── test.csv # 테스트 데이터
│
├── models/
│   ├── best_model.json # Best 모델 설계 정보
│   └── hyperparameter.json # 하이퍼파라미터 목록
│
├── scripts/
│   ├── data_train_options.sh # 학습 데이터 전처리 방식 입력
│   ├── data_test_options.sh # 예측 데이터 전처리 방식 입력
│   ├── model_options.sh # 모델 종류 및 경로 입력
│   └── train_options.sh # 모델 학습 관련 입력
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py # 데이터 전처리
│   ├── model.py # 모델 선택 또는 불러오기
│   ├── train.py # 모델 학습
│   ├── predict.py # 예측 및 결과 저장
│   └── utils.py # 유틸리티 함수
│
├── requirements.txt # 의존성
├── run_service.sh # 작업 스크립트
└── setup.sh # 가상환경 스크립트
```

## 모듈화 & 가상환경 기획

### 가상환경 실행
- `virtualenv` 를 사용해 파이썬 가상환경을 생성하고 활성화
- `requirements.txt` 파일을 참조하여 가상환경 생성 시 의존성을 설치
- 가상환경 경로를 입력으로 받아 해당 가상환경이 있으면 활성화, 아니면 해당 경로에 가상환경 생성
- 파이프라인을 종료하고 가상환경을 삭제할지는 추후에 판단

### 데이터 전처리
- 각 컬럼의 타입을 인식하여 별도의 명시 없이 범주형/수치형 컬럼을 구분
- 수치형 컬럼은 `StandardScaler` 로 인코딩하고 결측치는 중앙값으로 대체
- 범주형 컬럼은 `OrdinalEncoder` 로 인코딩하고 결측치는 최빈값으로 대체
- 고정 경로가 아닌, CSV 파일 경로를 입력으로 받아 데이터 읽어오기
- 가공한 데이터를 캐시처럼 별도로 저장하고 반복 학습 시 재사용할지는 추후에 판단

### 모델 선택
- `XGBClassifier`, `LogisticRegression` 두 가지 모델을 선택적으로 사용
   - 선정 사유는 대회 수상자 코드를 참고하여 주로 사용되는 모델을 선택
- 하이퍼파라미터 목록을 만들어두고 n번째 파라미터로 모델을 정의할 수 있게 파라미터 선택 기능 구현
- 모델 경로가 입력될 경우 해당 모델을 불러오기
- Best 모델의 설계 정보는 별도로 저장

### 모델 학습
- 미니 배치 학습 설계 (구현 실패)
- 모델 학습 후 AUC 평가 지표로 모델을 검증하여 Best 모델을 업데이트
   - 경진대회에서 평가하는 기준

### 예측 및 결과 저장
- Best 모델 또는 임의의 모델을 불러와 범주를 예측
- 예측 결과는 CSV 파일로 저장

## 모듈화 구현

### 데이터 전처리

```python
class Dataset:
    def __init__(self,
            file_path: Path,
            label_column: Optional[str] = None,
            drop_columns: Optional[Sequence[str]] = None,
            numeric_columns: Optional[Sequence[str]] = None,
            categorical_columns: Optional[Sequence[str]] = None,
            scaler: Optional[Literal["Standard","MinMax"]] = "Standard",
            test_size: float = 0.2,
            random_state: int = RANDOM_STATE,
            sep: str = ',',
            header: Optional[int] = 0,
            encoding = "utf-8",
            **kwargs
        ):
        self.file_path = file_path
        self.options = Options(sep, header, encoding, **kwargs)
        self.columns = Columns(label_column, drop_columns, numeric_columns, categorical_columns, sample_data=self.head())
        self._init_scaler(scaler)
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.test_size = test_size
        self.random_state = random_state

    def __len__(self) -> int:
        return read_nrows(self.file_path, self.options.get("header"))

    def __getitem__(self, locator=None) -> pd.DataFrame:
        if isinstance(locator, Tuple) and (len(locator) <= 2):
            return read_csv(self.file_path, *locator, **self.options)
        else: return read_csv(self.file_path, locator, **self.options)
```

- CSV 파일 경로 및 데이터 전처리와 관련된 정보를 `Dataset` 에서 관리
- `Dataset` 은 데이터를 불러오는 기능만 구현하고 데이터 객체를 직접 보유하지 않음
- 데이터를 불러올 때 행/열에 대한 2차원 슬라이싱을 지원

```python
class Options(dict):
    def __init__(self,
            sep: str = ',',
            header: Optional[int] = 0,
            encoding = "utf-8",
            **kwargs
        ):
        super().__init__(sep=sep, header=header, encoding=encoding, **kwargs)
```

- `Dataset` 의 입력 파라미터 중 CSV 형식과 관련된 파라미터는 `Options` 구조체로 정의
- `Options` 는 내장 딕셔너리를 그대로 상속받으며 딕셔너리 역할만 수행

```python
class Columns:
    def __init__(self,
            label: Optional[str] = None,
            drop: Optional[Sequence[str]] = None,
            numeric: Optional[Sequence[str]] = None,
            categorical: Optional[Sequence[str]] = None,
            sample_data: Optional[pd.DataFrame] = None):
        self.set_label_column(label, sample_data)
        self.set_drop_columns(drop, sample_data)
        self.set_numeric_columns(numeric, sample_data)
        self.set_categorical_columns(categorical, sample_data)
```

- `Dataset` 의 입력 파라미터 중에서 컬럼과 관련된 파라미터는 `Columns` 구조체로 정의
- `Columns` 는 샘플 데이터를 입력으로 받아 범주형/수치형 컬럼을 구분
   - 또는, `numeric`, `categorical` 입력을 받아 범주형/수치형 컬럼을 직접 지정
- `label` 에 열 명칭을 입력해 `train_test_split` 시 독립 변수와 종속 변수를 구분
- `drop` 에 입력된 열 명칭 목록은 모델 학습 시 제외

```python
class DataLoader:
    def __init__(self,
            dataset: Dataset,
            batch_size: int = 128,
            shuffle: bool = True
        ):
        X_train, X_val, y_train, y_val = dataset.get_dataset()
        self.train_dataset = (X_train, y_train)
        self.val_dataset = (X_val, y_val)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.toggle_dataset(split="train")
        self._reset_indices()

    def get_dataset(self) -> Tuple[pd.DataFrame,pd.Series]:
        return self.train_dataset if self.split == "train" else self.val_dataset
```

- `Dataset` 을 입력으로 받아 실제로 데이터를 불러오는 `DataLoader` 를 구현
- `train_test_split` 으로 학습/검증 데이터를 분리
- `toggle_dataset()` 을 통해 학습 또는 검증 중 하나의 데이터셋을 지정하고 `get_dataset()` 으로 해당하는 데이터셋을 반환
- 미니 배치 학습을 위해 `__iter__()`, `__next__()` 를 구현했지만, 짧은 시간에 경사 하강법을 적용하기 어려운 문제로 미사용

### 모델 선택

```python
def load_model_info(model_type: Literal["best","xgboost","logistic"]="best", model_path=str(), model_params: Dict=dict()) -> Dict:
    options = dict(model_type=model_type, model_path=model_path, model_params=model_params)
    if (model_type.lower() == "best") and os.path.exists(BEST_MODEL):
        return _load_best_model()
    elif model_type.lower() == "xgboost":
        params = _get_xgboost_params(**model_params)
        return dict(model=load_xgboost_model(model_path, **params), options=dict(options, model_params=params))
    elif model_type.lower() == "logistic":
        params = _get_logistic_params(**model_params)
        return dict(model=load_logistic_model(model_path, **params), options=dict(options, model_params=params))
    else: raise ValueError(f"지원하지 않는 모델 유형입니다: \"{model_type}\"")
```

- 모델 종류는 `Best 모델`, `XGBClassifier`, `LogisticRegression` 을 지원

```python
BEST_MODEL = MODEL_DIR / "best_model.json"

def _load_best_model() -> Dict:
    if os.path.exists(BEST_MODEL):
        with open(BEST_MODEL, 'r', encoding="utf-8") as file:
            options = json.loads(file.read())["model_options"]
            if options["model_type"] in ("xgboost","logistic"):
                return load_model_info(**options)
            else: return load_model_info("xgboost")
    else: raise ValueError("모델이 존재하지 않습니다.")
```

- Best 모델 선택 시 `models/best_model.json` 파일에서 모델 종류와 경로를 읽어서 `load_model_info()` 를 다시 호출

```python
HYPERPARAMETER = MODEL_DIR / "hyperparameter.json"

def _get_xgboost_params(random_state: int=RANDOM_STATE, eval_metric="auc", **params) -> Dict:
    params = params or select_hyperparameter(HYPERPARAMETER, model_type="xgboost")
    return dict(params, random_state=random_state, eval_metric=eval_metric)

def select_hyperparameter(file_path: Path, model_type: Literal["xgboost","logistic"]="xgboost",
                        page_size: int=10, start: int=0, num_preview=3, clear=True) -> Dict:
    with open(file_path, 'r', encoding="utf-8") as file:
        selections = json.loads(file.read())[model_type]
    while True:
        cur = selections[start:start+page_size]
        has_prev = (start > 0)
        has_next = (len(cur) == page_size)
        _print_selections(cur, start, num_preview, clear)
        selected = input(' '.join(filter(None, [
            ("[이전 페이지: prev]" if has_prev else ''),
            ("[다음 페이지: next]" if has_next else ''),
            "[선택: {}]".format(start if len(cur) == 1 else f"{start}-{start+len(cur)-1}")]))+' ')
        if (selected == "prev") and has_prev:
            start = start - page_size
        elif (selected == "next") and has_next:
            start = start + page_size
        elif selected.isdigit() and (int(selected) in range(start, start+page_size)):
            return selections[int(selected)]
        else: raise ValueError(f"입력이 올바르지 않습니다: \"{selected}\"")
```

- 하이퍼파라미터 선택 과정은 페이지 단위로 n번째 파라미터를 선택할 수 있게 구성
- `models/hyperparameter.json` 은 `{"xgboost": [{}], "logistic": [{}]}` 형식으로, 모델 종류에 대한 파라미터 배열이 입력되어 있음
- 페이지 단위로 파라미터를 읽으면서 특정 파라미터의 인덱스 번호를 입력하여 파라미터 선택
- 선택 화면에서 `num_preview` 만큼의 키값을 미리보기로 출력

### 모델 학습

```python
def train(dataset: Dataset, model: Model, model_type: Literal["xgboost","logistic"]="xgboost",
        epoch: int=1, batch_size: int=128, shuffle=True, **kwargs) -> Dict[str,float]:
    data_loader = DataLoader(dataset, batch_size, shuffle)
    X_train, X_val, y_train, y_val = data_loader.get_all_dataset()
    if model_type == "xgboost":
        return train_xgboost(model, X_train, X_val, y_train, y_val)
    elif model_type == "logistic":
        return train_logistic(model, X_train, X_val, y_train, y_val)
    else: raise ValueError(f"지원하지 않는 모델 유형입니다: \"{model_type}\"")

def train_xgboost(model: XGBClassifier,
                X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> Dict[str,float]:
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return evaluate(model, X_val, y_val)

def train_logistic(model: LogisticRegression,
                X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> Dict[str,float]:
    model.fit(X_train, y_train)
    return evaluate(model, X_val, y_val)
```

- 학습 과정은 단순하게 데이터를 `model.fit()` 에 전달
- 미니 배치 학습을 위해서는 경사 하강법을 지원하는 모델로 바꿔야 해서 기획한 것은 미구현

```python
def evaluate(model: Model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str,float]:
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
    print("F1 Score:", f1)
    print("AUC Score:", auc)
    return dict(f1_score=f1, auc_score=auc)
```

- 검증 시 F1 score와 AUC score를 계산해 반환
- Best 모델 저장 시 AUC score를 비교해 성능이 개선되었으면 모델 업데이트

### 예측 및 결과 저장

```python
def predict(dataset: Dataset, model: Model, return_type: Literal["label","proba"]="label") -> pd.DataFrame:
    raw_data = dataset.read()
    X_test, y_test = dataset.transform(raw_data), raw_data[["UID"]].copy()
    if return_type == "proba":
        proba = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)
        return pd.concat([y_test, proba], axis=1)
    else:
        y_test["Prediction"] = model.predict(X_test)
        return y_test
```

## 학습 파이프라인 설계

### 가상환경

```bash
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
                    pip install -r requirements.txt
# if문 종료 및 현재 가상환경 이름 출력
```

- 현재 가상환경 `$VIRTUAL_ENV` 이 있다면 스크립트 종료
- 가상환경 경로 `venv_path` 를 입력으로 받고, 입력이 없다면 기본값 `.venv` 사용
- 경로 내 가상환경이 있다면 활성화
- 경로 내 가상환경이 없다면 가상환경을 생성할지 확인받고, 생성 및 의존성 설치

### 파라미터 입력

```bash
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
    ...

model_options=$(cat <<EOF | tr -d '\n'
model_type=$model_type
&model_path=$model_path
&model_params=$model_params
EOF
)
```

- 학습 파이프라인 초기에 파라미터를 입력받는 부분을 `scripts/` 경로 아래 4개의 스크립트 파일로 구분
- `source` 명령어로 스크립트를 실행하여 전체 스크립트 환경에서 변수를 공유하도록 설정
   - `random_state` 는 `train_test_split` 및 모델의 하이퍼파라미터에 공통적으로 사용
- 위 예시는 `scripts/model_options.sh` 의 일부분으로, 파라미터 입력 시 기본값을 미리 설정하고, 사용자가 빈 값을 입력하면 기본값을 사용
- 입력받은 변수들은 URL에 사용하는 쿼리 스트링 형태로 변환하여 스크립트마다 `*_options` 변수에 저장

### 작업 실행

```bash
#!/bin/bash

source setup.sh

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

...

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
    ...
```

- 파라미터 입력 스크립트를 모두 실행하고 `src/train.py` 모듈을 실행하는 모델 학습 함수 등을 구현
- while 반복문을 돌면서 4개의 Mode (`학습`, `예측`, `학습 > 예측`, `종료`) 중 하나를 입력으로 받아 함수를 실행

## 실행 예시

### Best 모델

![best](https://raw.githubusercontent.com/minyeamer/mlops-study/refs/heads/main/.images/play-best.gif)

### XGBClassifier + 파라미터 선택

![xgboost](https://raw.githubusercontent.com/minyeamer/mlops-study/refs/heads/main/.images/play-xgboost.gif)
