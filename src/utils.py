from typing import Callable, List, Optional, Sequence, Union
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


def read_csv(file_path: str, rows: Optional[Union[slice,Sequence[int]]]=None, cols: Optional[Union[slice,Sequence[int]]]=None,
            sep: str=',', header: Optional[int]=0, encoding="utf-8", **kwargs) -> pd.DataFrame:
    if rows is None:
        if cols is not None:
            cols = _slice_csv_by_cols(file_path, cols, sep, encoding) if isinstance(cols, slice) else cols
        return pd.read_csv(file_path, sep=sep, header=header, usecols=cols, encoding=encoding, **kwargs)
    elif isinstance(rows, slice):
        return _read_continuous_csv_rows(file_path, rows, cols, sep, header, encoding, **kwargs)
    elif isinstance(rows, Sequence):
        return _read_discrete_csv_rows(file_path, rows, cols, sep, header, encoding, **kwargs)
    else: raise InvalidIndexError((rows, cols))


def _read_continuous_csv_rows(file_path: str, rows: slice, cols: Optional[Union[slice,Sequence[int]]]=None,
                            sep: str=',', header: Optional[int]=0, encoding="utf-8", **kwargs) -> pd.DataFrame:
    rows = _slice_csv_by_rows(file_path, rows, header)
    row_options = dict(skiprows=range((header or 0), rows.start+(header or 0)), nrows=(rows.stop-rows.start))
    cols = _slice_csv_by_cols(file_path, cols, sep, encoding) if isinstance(cols, slice) else cols
    col_options = dict(usecols=cols)
    return pd.read_csv(file_path, sep=sep, header=header, **row_options, **col_options, encoding=encoding, **kwargs)


def _read_discrete_csv_rows(file_path: str, rows: Sequence[int], cols: Optional[Union[slice,Sequence[int]]]=None,
                            sep: str=',', header: Optional[int]=0, encoding="utf-8", **kwargs) -> pd.DataFrame:
    loc = _make_discrete_locator(cols)
    with open(file_path, 'r', encoding=encoding) as file:
        rows, header = list(), None
        for __i, __row in enumerate(csv.reader(file, delimiter=sep)):
            if __i == header: header = loc(__row)
            elif __i in rows: rows.append(loc(__row))
            else: pass
        return pd.DataFrame(rows, columns=header, **kwargs)


def _make_discrete_locator(cols: Optional[Union[slice,Sequence[int]]]=None) -> Callable[[List[str]],List[str]]:
    if isinstance(cols, slice):
        return (lambda row: row[cols])
    elif isinstance(cols, Sequence):
        return (lambda row: [__cell for __i, __cell in enumerate(row) if __i in cols])
    else: return (lambda row: row)


def _slice_csv_by_rows(file_path: str, rows: slice, header: Optional[int]=0) -> range:
    nrows = read_nrows(file_path, header) if ((rows.start or 0) < 0) or ((rows.stop or 0) <= 0) else 0
    start = (nrows + x) if (x := (rows.start or 0)) < 0 else x
    stop = (nrows + x) if (x := (rows.stop or nrows)) < 0 else x
    return range(start, stop, 1)


def _slice_csv_by_cols(file_path: str, cols: slice, sep: str=',', encoding="utf-8") -> range:
    start = cols.start or 0
    stop = cols.stop or read_ncols(file_path, sep, encoding)
    return range(start, stop)
