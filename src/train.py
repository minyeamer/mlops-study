from src import DATA_DIR, MODEL_DIR
from src.preprocess import Dataset
from src.model import get_xgboost_model, get_logistic_model

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import json

from typing import Dict, Literal, Union
import pandas as pd


Model = Union[XGBClassifier, LogisticRegression]


def train_xgboost(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> XGBClassifier:
    model = get_xgboost_model()
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    model.save_model(MODEL_DIR / "xgb_model.json")
    return model


def train_logistic(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> LogisticRegression:
    model = get_logistic_model()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / "logistic_model.joblib")
    return model


def save_metrics(metrics, model_type: Literal["xgboost","logistic"]):
    metrics_file = MODEL_DIR / "model_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[model_type] = metrics

    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)


def evaluate_model(model: Model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str,float]:
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    print("F1 Score:", f1)
    print("AUC Score:", auc)

    return {
        "f1_score": f1,
        "auc_score": auc
    }


def get_best_model_type() -> Model:
    metrics_file = MODEL_DIR / "model_metrics.json"
    if not metrics_file.exists():
        return "xgboost"

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    best_model = max(metrics.items(), key=lambda x: x[1]['auc_score'])
    return best_model[0]


def train(model_type: Literal["xgboost","logistic"]="xgboost"):
    X_train, X_val, y_train, y_val = Dataset(
        file_path=DATA_DIR / "train.csv",
        label_column="채무 불이행 여부",
        drop_columns=["UID"],
        test_size=0.2,
        random_state=42,
    ).load_dataset()

    if model_type == "xgboost":
        model = train_xgboost(X_train, X_val, y_train, y_val)
    elif model_type == "logistic":
        model = train_logistic(X_train, X_val, y_train, y_val)
    else:
        raise ValueError(f"Unknown model type: \"{model_type}\"")

    print(f"\nEvaluating \"{model_type}\" model:")
    metrics = evaluate_model(model, X_val, y_val)
    save_metrics(metrics, model_type)

    best_model_type = get_best_model_type()
    with open(MODEL_DIR / "best_model.txt", 'w') as f:
        f.write(best_model_type)
