from src.utils import read_nrows, read_csv

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from typing import Literal, Optional, Sequence, Tuple
from pathlib import Path


class Columns:
    def __init__(self,
            label: str,
            drop: Optional[Sequence[str]] = None,
            numeric: Optional[Sequence[str]] = None,
            categorical: Optional[Sequence[str]] = None,
            sample_data: Optional[pd.DataFrame] = None):
        self.set_label_column(label, sample_data)
        self.set_drop_columns(drop, sample_data)
        self.set_numeric_columns(numeric, sample_data)
        self.set_categorical_columns(categorical, sample_data)

    def set_label_column(self, column: str, sample_data: Optional[pd.DataFrame]=None):
        if (isinstance(sample_data, pd.DataFrame) and (column not in sample_data)) or (not column):
            raise KeyError(str(column))
        else: self.label = column

    def set_drop_columns(self, columns: Optional[Sequence[str]]=None, sample_data: Optional[pd.DataFrame]=None):
        if (columns is None) and isinstance(sample_data, pd.DataFrame):
            self.drop = [__col for __col in columns if __col in sample_data]
        else: self.drop = columns or list()

    def set_numeric_columns(self, columns: Optional[Sequence[str]]=None, sample_data: Optional[pd.DataFrame]=None):
        if (columns is None) and isinstance(sample_data, pd.DataFrame):
            columns = sample_data.select_dtypes(include=["int64","float64"]).columns.tolist()
        else: columns = columns or list()
        self.numeric = [__col for __col in columns if (__col != self.label) and (__col not in self.drop)]

    def set_categorical_columns(self, columns: Optional[Sequence[str]]=None, sample_data: Optional[pd.DataFrame]=None):
        if (columns is None) and isinstance(sample_data, pd.DataFrame):
            columns = sample_data.select_dtypes(include=["object"]).columns.tolist()
        else: columns = columns or list()
        self.categorical = [__col for __col in columns if (__col != self.label) and (__col not in self.drop)]


class Options(dict):
    def __init__(self,
            sep: str = ',',
            header: Optional[int] = 0,
            encoding = "utf-8",
            **kwargs):
        super().__init__(sep=sep, header=header, encoding=encoding, **kwargs)


class Dataset:
    def __init__(self,
            file_path: Path,
            label_column: str,
            drop_columns: Optional[Sequence[str]] = None,
            numeric_columns: Optional[Sequence[str]] = None,
            categorical_columns: Optional[Sequence[str]] = None,
            scaler: Optional[Literal["Standard","MinMax"]] = "Standard",
            test_size: float = 0.2,
            random_state: int = 42,
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

    def _init_scaler(self, scaler: Optional[Literal["Standard","MinMax"]]="Standard"):
        if scaler.lower() == "standard":
            self.scaler = StandardScaler()
        elif scaler.lower() == "minmax":
            self.scaler = MinMaxScaler()
        else: self.scaler = None

    def load_data(self) -> Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
        data = self[:,:]
        X = data.drop(columns=[self.columns.label])
        y = data[self.columns.label]
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
        missing_cols = (set(self.columns.numeric) | set(self.columns.categorical)) - set(data.columns)
        for __col in missing_cols:
            data[__col] = None
        data = self._drop_columns(data)
        data = self._handle_missing_values(data)
        data = self._encode_categorical_features(data, fit=False)
        data = self._scale_numerical_features(data, fit=False)
        return data

    def _drop_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.columns.drop:
            return data.drop(columns=self.columns.drop)
        else: return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        for __col in self.columns.numeric:
            if __col in data:
                data[__col] = data[__col].fillna(data[__col].median())
        for __col in self.columns.categorical:
            if __col in data:
                data[__col] = data[__col].fillna(data[__col].mode()[0])
        return data

    def _encode_categorical_features(self, data: pd.DataFrame, fit=True):
        if self.columns.categorical and (self.encoder is not None):
            if fit:
                data[self.columns.categorical] = self.encoder.fit_transform(data[self.columns.categorical])
            else:
                data[self.columns.categorical] = self.encoder.transform(data[self.columns.categorical])
        return data

    def _scale_numerical_features(self, data: pd.DataFrame, fit=True):
        if self.columns.numeric and (self.scaler is not None):
            if fit:
                data[self.columns.numeric] = self.scaler.fit_transform(data[self.columns.numeric])
            else:
                data[self.columns.numeric] = self.scaler.transform(data[self.columns.numeric])
        return data

    def __len__(self) -> int:
        return read_nrows(self.file_path, self.options.get("header"))

    def __getitem__(self, locator=None) -> pd.DataFrame:
        if isinstance(locator, Tuple) and (len(locator) <= 2):
            return read_csv(self.file_path, *locator, **self.options)
        else: return read_csv(self.file_path, locator, **self.options)

    def head(self, n: int=5) -> pd.DataFrame:
        return self[:n]

    def tail(self, n: int=5) -> pd.DataFrame:
        return self[n:]


class DataLoader:
    def __init__(self,
            dataset: Dataset,
            batch_size: int = 32,
            shuffle: bool = True):
        X_train, X_val, y_train, y_val = dataset.load_data()
        self.train_dataset = (X_train, y_train)
        self.val_dataset = (X_val, y_val)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.toggle_dataset(split="train")
        self._reset_indices()

    def toggle_dataset(self, split: Optional[Literal["train","val"]]=None):
        toggle = {"train": "val", "val": "train"}
        try: self.split = split if split in toggle else toggle[self.split]
        except: self.split = "train"
        self.indices = np.arange(len(self.select_dataset()[1]))

    def select_dataset(self) -> Tuple[pd.DataFrame,pd.Series]:
        return self.train_dataset if self.split == "train" else self.val_dataset

    def _reset_indices(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current = 0

    def __iter__(self):
        self._reset_indices()
        return self

    def __next__(self):
        X, y = self.select_dataset()
        if self.current >= len(y):
            raise StopIteration
        indices = self.indices[self.current:self.current+self.batch_size]
        batch = X.iloc[indices], y.iloc[indices]
        self.current += self.batch_size
        return batch
