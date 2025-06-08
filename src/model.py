from src import MODEL_DIR, BEST_MODEL, RANDOM_STATE

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import joblib

from typing import Dict, Literal, Union
from pathlib import Path
import json
import os


HYPERPARAMETER = MODEL_DIR / "hyperparameter.json"

Model = Union[XGBClassifier, LogisticRegression]


def load_model(model_type: Literal["best","xgboost","logistic"]="best", model_path=str(), model_params: Dict=dict()) -> Model:
    return load_model_info(model_type, model_path, model_params)["model"]


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


def _load_best_model() -> Dict:
    if os.path.exists(BEST_MODEL):
        with open(BEST_MODEL, 'r', encoding="utf-8") as file:
            options = json.loads(file.read())["model_options"]
            if options["model_type"] in ("xgboost","logistic"):
                return load_model_info(**options)
            else: return load_model_info("xgboost")
    else: raise ValueError("모델이 존재하지 않습니다.")


def load_xgboost_model(model_path=str(), **params) -> XGBClassifier:
    model = XGBClassifier(**params)
    if model_path and os.path.exists(model_path):
        model.load_model(model_path)
    return model


def _get_xgboost_params(random_state: int=RANDOM_STATE, eval_metric="auc", **params) -> Dict:
    params = params or select_hyperparameter(HYPERPARAMETER, model_type="xgboost")
    return dict(params, random_state=random_state, eval_metric=eval_metric)


def load_logistic_model(model_path=str(), **params) -> LogisticRegression:
    if model_path and os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return LogisticRegression(**params)


def _get_logistic_params(random_state: int=RANDOM_STATE, **params) -> Dict:
    params = params or select_hyperparameter(HYPERPARAMETER, model_type="xgboost")
    return dict(params, random_state=random_state)


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


def _print_selections(selections: Dict, start: int=0, num_preview=3, clear=True):
    if clear: os.system("clear")
    def _preview(params: Dict) -> str:
        items = list(params.items())
        return json.dumps(dict(items[:num_preview], **({"...": "..."} if len(items) > num_preview else {})))
    for __i, __params in enumerate(selections, start=start):
        print(f"({__i}) {_preview(__params)}")
