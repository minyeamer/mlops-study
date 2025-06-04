from src.utils import read_nrows, read_csv

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

from typing import Literal, Optional, Sequence, Tuple
from collections import defaultdict
from pathlib import Path


class Dataset:
    def __init__(self,
            file_path: Path,
            label_column: str,
            sep: str=',',
            header: Optional[int]=0,
            encoding="utf-8",
            numeric_columns: Optional[Sequence[str]]=None,
            categorical_columns: Optional[Sequence[str]]=None,
            drop_columns: Optional[Sequence[str]]=None,
            scaler: Optional[Literal["Standard","MinMax"]]="Standard",
            test_size: float=0.2,
            random_state: int=42,
            **kwargs
        ):
        self.file_path = file_path
        self.read_info = dict(sep=sep, header=header, encoding=encoding, **kwargs)
        self._init_columns(label_column, numeric_columns, categorical_columns, drop_columns)
        self.label_encoders = defaultdict(LabelEncoder)
        self._init_scaler(scaler)
        self.test_size = test_size
        self.random_state = random_state

    def _init_columns(self, label: str,
                    numeric: Optional[Sequence[str]]=None,
                    categorical: Optional[Sequence[str]]=None,
                    drop: Optional[Sequence[str]]=None):
        data = self.head().drop(columns=[label]+(drop or list()))
        if numeric is None:
            numeric = data.select_dtypes(include=["int64","float64"]).columns.tolist()
        if categorical is None:
            categorical = data.select_dtypes(include=["object"]).columns.tolist()
        self.column_info = dict(label=label, numeric=numeric, categorical=categorical, drop=drop)

    def _init_scaler(self, scaler: Optional[Literal["Standard","MinMax"]]="Standard"):
        if scaler.lower() == "standard":
            self.scaler = StandardScaler()
        elif scaler.lower() == "minmax":
            self.scaler = MinMaxScaler()
        else: self.scaler = None

    def load_dataset(self) -> Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
        data = self[:,:]
        X = data.drop(columns=[self.column_info["label"]])
        y = data[self.column_info["label"]]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        X_train = self.fit_transform(X_train)
        X_val = self.transform(X_val)
        return X_train, X_val, y_train, y_val

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._drop_columns(data)
        data = self._handle_missing_values(data)
        data = self._encode_categorical_features(data, fit=True)
        data = self._scale_numerical_features(data, fit=True)
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        missing_cols = (set(self.column_info["numeric"]) | set(self.column_info["categorical"])) - set(data.columns)
        for __col in missing_cols:
            data[__col] = None
        data = self._drop_columns(data)
        data = self._handle_missing_values(data)
        data = self._encode_categorical_features(data, fit=False)
        data = self._scale_numerical_features(data, fit=False)
        return data

    def _drop_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column_info["drop"] is not None:
            return data.drop(columns=self.column_info["drop"])
        else: return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        for __col in self.column_info["numeric"]:
            if __col in data:
                data[__col] = data[__col].fillna(data[__col].median())
        for __col in self.column_info["categorical"]:
            if __col in data:
                data[__col] = data[__col].fillna(data[__col].mode()[0])
        return data

    def _encode_categorical_features(self, data: pd.DataFrame, fit=True):
        for __col in self.column_info["categorical"]:
            if __col in data.columns:
                if fit:
                    data[__col] = self.label_encoders[__col].fit_transform(data[__col])
                else:
                    unique_values = set(data[__col].unique())
                    known_values = set(self.label_encoders[__col].classes_)
                    unknown_values = unique_values - known_values
                    if unknown_values:
                        data.loc[data[__col].isin(unknown_values), __col] = self.label_encoders[__col].classes_[0]
                    data[__col] = self.label_encoders[__col].transform(data[__col])
        return data

    def _scale_numerical_features(self, data: pd.DataFrame, fit=True):
        if self.column_info["numeric"]:
            numerical_data = data[self.column_info["numeric"]]
            if fit:
                data[self.column_info["numeric"]] = self.scaler.fit_transform(numerical_data)
            else:
                data[self.column_info["numeric"]] = self.scaler.transform(numerical_data)
        return data

    def __len__(self) -> int:
        return read_nrows(self.file_path, self.read_info["header"])

    def __getitem__(self, locator=slice(None)) -> pd.DataFrame:
        if isinstance(locator, Tuple) and (len(locator) <= 2):
            return read_csv(self.file_path, *locator, **self.read_info)
        else: return read_csv(self.file_path, locator, **self.read_info)

    def head(self, n: int=5) -> pd.DataFrame:
        return self[:n]

    def tail(self, n: int=5) -> pd.DataFrame:
        return self[n:]


class DataLoader:
    ...
