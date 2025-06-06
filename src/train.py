from src import DATA_DIR, MODEL_DIR, BEST_MODEL, RANDOM_STATE
from src.preprocess import Dataset, DataLoader
from src.model import Model, load_model
from src.utils import safe_int, safe_float, safe_bool

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import json

from typing import Any, Dict, Literal, Tuple
import argparse
import pandas as pd


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


def train(dataset: Dataset, model: Model, epoch: int=1, batch_size: int=128, shuffle=True, **kwargs):
    data_loader = DataLoader(dataset, batch_size, shuffle)
    for __epoch in range(epoch):
        for __i, (__xb, __yb) in enumerate(data_loader):
            print(f"Epoch: {__epoch}, Batch: {__i}, Size: {len(__yb)}")
            model.fit(__xb, __yb)

#     if model_type == "xgboost":
#         model = train_xgboost(X_train, X_val, y_train, y_val)
#     elif model_type == "logistic":
#         model = train_logistic(X_train, X_val, y_train, y_val)
#     else:
#         raise ValueError(f"Unknown model type: \"{model_type}\"")

#     print(f"\nEvaluating \"{model_type}\" model:")
#     metrics = evaluate_model(model, X_val, y_val)
#     save_metrics(metrics, model_type)

#     best_model_type = get_best_model_type()
#     with open(MODEL_DIR / "best_model.txt", 'w') as f:
#         f.write(best_model_type)



# def train_xgboost(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> XGBClassifier:
#     model = load_xgboost_model()
#     eval_set = [(X_train, y_train), (X_val, y_val)]
#     model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
#     model.save_model(MODEL_DIR / "xgb_model.json")
#     return model


# def train_logistic(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> LogisticRegression:
#     model = load_logistic_model()
#     model.fit(X_train, y_train)
#     joblib.dump(model, MODEL_DIR / "logistic_model.joblib")
#     return model


# def save_metrics(metrics, model_type: Literal["xgboost","logistic"]):
#     metrics_file = MODEL_DIR / "model_metrics.json"
#     if metrics_file.exists():
#         with open(metrics_file, 'r') as f:
#             all_metrics = json.load(f)
#     else:
#         all_metrics = {}

#     all_metrics[model_type] = metrics

#     with open(metrics_file, 'w') as f:
#         json.dump(all_metrics, f, indent=4)


# def evaluate_model(model: Model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str,float]:
#     y_pred = model.predict(X_val)
#     f1 = f1_score(y_val, y_pred)
#     auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

#     print("F1 Score:", f1)
#     print("AUC Score:", auc)

#     return {
#         "f1_score": f1,
#         "auc_score": auc
#     }


# def get_best_model_type() -> Model:
#     metrics_file = MODEL_DIR / "model_metrics.json"
#     if not metrics_file.exists():
#         return "xgboost"

#     with open(metrics_file, 'r') as f:
#         metrics = json.load(f)

#     best_model = max(metrics.items(), key=lambda x: x[1]['auc_score'])
#     return best_model[0]



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
            model_params: str = str(),
            random_state: str = str(RANDOM_STATE),
            verbose: str = "0"
        ):
        super().__init__(
            model_type = (model_type if model_type.lower() in ("best","xgboost","logistic") else "best"),
            model_path = (model_path.format(MODEL_DIR=MODEL_DIR) if "{MODEL_DIR}" in model_path else model_path),
            model_params = (_parse_params(model_params, cast=True) if model_params else dict()),
            random_state = safe_int(random_state, default=RANDOM_STATE),
            verbose = safe_int(verbose, default=0)
        )


class TrainOptions(dict):
    def __init__(self,
            epoch: str = "1",
            batch_size: str = "32",
            shuffle: str = "true"
        ):
        super().__init__(
            epoch = safe_int(epoch, default=1),
            batch_size = safe_int(batch_size, default=32),
            shuffle = safe_bool(shuffle, default=True)
        )


def _parse_args() -> Dict[str,Dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="데이터 전처리 옵션")
    parser.add_argument("--model", type=str, required=True, help="모델 초기화 옵션")
    parser.add_argument("--train", type=str, required=True, help="모델 학습 옵션")

    args = parser.parse_args()
    return dict(
        data_options=DataOptions(**_parse_params(args.data, cast=False)),
        model_options=ModelOptions(**_parse_params(args.model, cast=False)),
        train_options=TrainOptions(**_parse_params(args.train, cast=False))
    )


def _parse_params(params: str, cast=False) -> Dict:
    def type_cast(key: str, value: str) -> Tuple[str,Any]:
        if cast:
            for safe_cast in [safe_int, safe_float, safe_bool]:
                __value = safe_cast(value, default=None)
                if __value is not None:
                    return key, __value
        return key, value
    return dict(filter(None, [type_cast(*__kv.split('=', maxsplit=1)) for __kv in params.split('&')]))


def main(data_options: Dict, model_options: Dict, train_options: Dict):
    dataset = Dataset(**data_options)
    model = load_model(**model_options)
    train(dataset, model, **train_options)


if __name__ == "__main__":
    main(**_parse_args())
