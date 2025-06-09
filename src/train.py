from src import DATA_DIR, MODEL_DIR, BEST_MODEL, RANDOM_STATE
from src.preprocess import Dataset, DataLoader
from src.model import Model, load_model_info
from src.utils import safe_int, safe_float, safe_bool, parse_params

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import json
import pandas as pd

from typing import Dict, Literal
from pathlib import Path
import argparse


DEFAULT_TRAIN_PRAMS = {
    "label_column": "채무 불이행 여부",
    "drop_columns": ["UID"],
    "test_size": 0.2,
    "random_state": 42
}

DEFAULT_MODEL_PARAMS = {
    "model_type": "xgboost",
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.15,
    "random_state": 42,
    "eval_metric": "auc"
}


###################################################################
############################## Train ##############################
###################################################################


def train(dataset: Dataset, model: Model, model_type: Literal["xgboost","logistic"]="xgboost",
        epoch: int=1, batch_size: int=128, shuffle=True, **kwargs) -> Dict[str,float]:
    data_loader = DataLoader(dataset, batch_size, shuffle)
    # for __epoch in range(epoch):
    #     for __i, (__xb, __yb) in enumerate(data_loader):
    #         print(f"Epoch: {__epoch}, Batch: {__i}, Size: {len(__yb)}")
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


def evaluate(model: Model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str,float]:
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
    print("F1 Score:", f1)
    print("AUC Score:", auc)
    return dict(f1_score=f1, auc_score=auc)


###################################################################
############################### Save ##############################
###################################################################

XGBOOST_MODEL = MODEL_DIR / "xgb_model.json"
LOGISTIC_MODEL = MODEL_DIR / "logistic_model.joblib"


def save_model(model: Model, metrics: Dict[str,float], model_type: Literal["xgboost","logistic"]="xgboost",
            save_mode: Literal["best_only","overwrite","primary"]="best_only", **kwargs) -> Dict:
    model_path = get_default_model_path(model_type)
    is_best = is_best_model(metrics["auc_score"])

    if (save_mode == "primary") and not model_path.exists():
        save = True
    elif (save_mode == "overwrite") or ((save_mode == "best_only") and is_best):
        save = True
    else: save = False

    if save:
        if model_type == "xgboost":
            save_xgboost(model, model_path)
        else: save_logistic(model, model_path)
    return dict(model_path=model_path, is_best=is_best, is_saved=save)


def save_xgboost(model: XGBClassifier, save_to: Path):
    model.save_model(save_to)
    print("XGBClassifier 모델의 파라미터를 업데이트했습니다.")


def save_logistic(model: LogisticRegression, save_to: Path):
    joblib.dump(model, save_to)
    print("LogisticRegression 모델의 파라미터를 업데이트했습니다.")


def get_default_model_path(model_type: Literal["xgboost","logistic"]="xgboost") -> Path:
    if model_type == "xgboost":
        return XGBOOST_MODEL
    elif model_type == "logistic":
        return LOGISTIC_MODEL
    else: raise ValueError(f"지원하지 않는 모델 유형입니다: \"{model_type}\"")


def is_best_model(auc_score: float) -> bool:
    if BEST_MODEL.exists():
        with open(BEST_MODEL, 'r', encoding="utf-8") as file:
            best_score = json.loads(file.read())["metrics"]["auc_score"]
            is_best = auc_score > best_score
            if is_best:
                print(f"AUC Score가 갱신되었습니다: {round(best_score,5)} > {round(auc_score,5)}")
            return is_best
    else: return True


###################################################################
############################### Main ##############################
###################################################################

class DataOptions(dict):
    def __init__(self,
            file_path: str,
            label_column: str,
            drop_columns: str = str(),
            numeric_columns: str = str(),
            categorical_columns: str = str(),
            scaler: str = "Standard",
            test_size: str = "0.2",
            random_state: str = str(RANDOM_STATE)
        ):
        super().__init__(
            file_path = (file_path.format(DATA_DIR=DATA_DIR) if "{DATA_DIR}" in file_path else file_path),
            label_column = label_column,
            drop_columns = (drop_columns.split(',') if drop_columns else None),
            numeric_columns = (numeric_columns.split(',') if numeric_columns else None),
            categorical_columns = (categorical_columns.split(',') if categorical_columns else None),
            scaler = (scaler if scaler.lower() in ("standard","minmax") else "Standard"),
            test_size = safe_float(test_size, default=0.2),
            random_state = safe_int(random_state, default=RANDOM_STATE)
        )


class ModelOptions(dict):
    def __init__(self,
            model_type: str,
            model_path: str = str(),
            model_params: str = str()
        ):
        super().__init__(
            model_type = (model_type if model_type.lower() in ("best","xgboost","logistic") else "best"),
            model_path = (model_path.format(MODEL_DIR=MODEL_DIR) if "{MODEL_DIR}" in model_path else model_path),
            model_params = (parse_params(model_params, cast=True) if model_params else dict())
        )


class TrainOptions(dict):
    def __init__(self,
            epoch: str = "1",
            batch_size: str = "128",
            shuffle: str = "true"
        ):
        super().__init__(
            epoch = safe_int(epoch, default=1),
            batch_size = safe_int(batch_size, default=128),
            shuffle = safe_bool(shuffle, default=True)
        )


def _parse_args() -> Dict[str,Dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="데이터 전처리 옵션")
    parser.add_argument("--model", type=str, required=True, help="모델 초기화 옵션")
    parser.add_argument("--train", type=str, required=True, help="모델 학습 옵션")

    args = parser.parse_args()
    return dict(
        data_options=DataOptions(**parse_params(args.data, cast=False)),
        model_options=ModelOptions(**parse_params(args.model, cast=False)),
        train_options=TrainOptions(**parse_params(args.train, cast=False))
    )


def save_best_options(data_options: Dict, model_options: Dict, train_options: Dict, metrics: Dict):
    with open(BEST_MODEL, 'w', encoding="utf-8") as file:
        options = dict(
            data_options = data_options,
            model_options = model_options,
            train_options = train_options,
            metrics = metrics
        )
        json.dump(options, file, indent=2, ensure_ascii=False)


def main(data_options: Dict, model_options: Dict, train_options: Dict):
    dataset = Dataset(**data_options)
    model_info = load_model_info(**model_options)
    model, model_options = model_info["model"], model_info["options"]
    metrics = train(dataset, model, model_options["model_type"], **train_options)
    status = save_model(model, metrics, model_options["model_type"], **train_options)
    if status["is_best"] and status["is_saved"]:
        model_options["model_path"] = str(status["model_path"])
        save_best_options(data_options, model_options, train_options, metrics)


if __name__ == "__main__":
    main(**_parse_args())
