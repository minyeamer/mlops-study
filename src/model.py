from typing import Dict, Literal
from pathlib import Path
import json
import os


MODEL_DIR = Path("models")
HYPERPARAMETER = MODEL_DIR / "hyperparameter.json"


def get_xgboost_model(**params):
    from xgboost import XGBClassifier
    return XGBClassifier(**(params or select_hyperparameter(HYPERPARAMETER, model_type="xgboost")))


def get_logistic_model(**params):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**(params or select_hyperparameter(HYPERPARAMETER, model_type="logistic")))


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
