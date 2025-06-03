from typing import Optional, Sequence, Union
from pandas.errors import InvalidIndexError
import pandas as pd
import csv


def read_nrows(file_path: str, header: Optional[int]=0) -> int:
        has_header = isinstance(header, int)
        with open(file_path, "rb") as file:
            return sum(1 for _ in file) - int(has_header)


def read_ncols(file_path: str, sep: str=',', encoding="utf-8") -> int:
        with open(file_path, 'r', encoding=encoding) as file:
            return len(file.readline().split(sep))


def read_csv(file_path: str, rows: Union[slice,Sequence[int]]=slice(None), cols: Union[slice,Sequence[int]]=slice(None),
            sep: str=',', header: Optional[int]=0, encoding="utf-8") -> pd.DataFrame:
    if isinstance(rows, slice):
        return _read_continuous_csv_rows(file_path, rows, cols, sep, header, encoding)
    elif isinstance(rows, Sequence):
        return _read_discrete_csv_rows(file_path, rows, cols, sep, header, encoding)
    else: raise InvalidIndexError((rows, cols))


def _read_continuous_csv_rows(file_path: str, rows: slice, cols: Union[slice,Sequence[int]],
                            sep: str=',', header: Optional[int]=0, encoding="utf-8") -> pd.DataFrame:
    rows = _slice_csv_by_rows(file_path, rows, header)
    skiprows = range((header or 0), rows.start+(header or 0))
    cols = _slice_csv_by_cols(file_path, cols, sep, encoding) if isinstance(cols, slice) else cols
    return pd.read_csv(file_path, sep=sep, header=header, usecols=cols, skiprows=skiprows, nrows=(rows.stop-rows.start), encoding=encoding)


def _read_discrete_csv_rows(file_path: str, rows: Sequence[int], cols: Union[slice,Sequence[int]],
                            sep: str=',', header: Optional[int]=0, encoding="utf-8") -> pd.DataFrame:
    def loc(row: Sequence, cols: Union[slice,Sequence[int]]) -> Sequence:
        if isinstance(cols, Sequence):
            return [__cell for __i, __cell in enumerate(row) if __i in cols]
        else: return row[cols]
    with open(file_path, 'r', encoding=encoding) as file:
        rows, header = list(), None
        for __i, __row in enumerate(csv.reader(file, delimiter=sep)):
            if __i == header: header = loc(__row, cols)
            elif __i in rows: rows.append(loc(__row, cols))
            else: pass
        return pd.DataFrame(rows, columns=header)


def _slice_csv_by_rows(file_path: str, rows: slice, header: Optional[int]=0) -> range:
    nrows = read_nrows(file_path, header) if ((rows.start or 0) < 0) or ((rows.stop or 0) <= 0) else 0
    start = (nrows + x) if (x := (rows.start or 0)) < 0 else x
    stop = (nrows + x) if (x := (rows.stop or nrows)) < 0 else x
    return range(start, stop, 1)


def _slice_csv_by_cols(file_path: str, cols: slice, sep: str=',', encoding="utf-8") -> range:
    start = cols.start or 0
    stop = cols.stop or read_ncols(file_path, sep, encoding)
    return range(start, stop)
