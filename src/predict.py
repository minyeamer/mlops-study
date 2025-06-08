from src import DATA_DIR, MODEL_DIR
from src.preprocess import Dataset
from src.model import Model, load_model
from src.utils import parse_params

from typing import Dict, Literal
import argparse
import pandas as pd


def predict(dataset: Dataset, model: Model, return_type: Literal["label","proba"]="label") -> pd.DataFrame:
    raw_data = dataset.read()
    X_test, y_test = dataset.transform(raw_data), raw_data[["UID"]].copy()
    if return_type == "proba":
        proba = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)
        return pd.concat([y_test, proba], axis=1)
    else:
        y_test["Prediction"] = model.predict(X_test)
        return y_test


###################################################################
############################### Main ##############################
###################################################################

class DataOptions(dict):
    def __init__(self,
            file_path: str,
            drop_columns: str = str(),
            numeric_columns: str = str(),
            categorical_columns: str = str(),
            scaler: str = "Standard",
            **kwargs
        ):
        super().__init__(
            file_path = (file_path.format(DATA_DIR=DATA_DIR) if "{DATA_DIR}" in file_path else file_path),
            drop_columns = (drop_columns.split(',') if drop_columns else None),
            numeric_columns = (numeric_columns.split(',') if numeric_columns else None),
            categorical_columns = (categorical_columns.split(',') if categorical_columns else None),
            scaler = (scaler if scaler.lower() in ("standard","minmax") else "Standard")
        )


class ModelOptions(dict):
    def __init__(self,
            model_type: str,
            model_path: str = str(),
            **kwargs
        ):
        super().__init__(
            model_type = (model_type if model_type.lower() in ("best","xgboost","logistic") else "best"),
            model_path = (model_path.format(MODEL_DIR=MODEL_DIR) if "{MODEL_DIR}" in model_path else model_path)
        )


def _parse_args() -> Dict[str,Dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="데이터 전처리 옵션")
    parser.add_argument("--model", type=str, required=True, help="모델 초기화 옵션")
    parser.add_argument("--return_type", type=str, required=False, help="예측 결과 유형", default="label")
    parser.add_argument("--save_to", type=str, required=False, help="예측 파일 저장", default=str())

    args = parser.parse_args()
    return dict(
        data_options=DataOptions(**parse_params(args.data, cast=False)),
        model_options=ModelOptions(**parse_params(args.model, cast=False)),
        save_to=args.save
    )


def main(data_options: Dict, model_options: Dict, return_type: Literal["label","proba"]="label", save_to=str()) -> pd.DataFrame:
    dataset = Dataset(**data_options)
    dataset.load_encoder()
    dataset.load_scaler()
    model = load_model(**model_options)
    preds = predict(dataset, model, return_type)
    if save_to:
        preds.to_csv(save_to, index=False)
    return preds


if __name__ == "__main__":
    main(**_parse_args())
