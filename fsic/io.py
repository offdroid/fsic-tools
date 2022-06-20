import json
import typing as ty
from pathlib import Path

import pandas as pd


def _df_has_required_columns(df: pd.DataFrame) -> bool:
    return all(
        map(
            lambda x: x in set(df.columns).union(df.index.names),
            ["timesteps", "length", "reward", "failure", "origin", "uid"],
        )
    )


def save(df: pd.DataFrame, resolution: ty.Dict, path):
    if not _df_has_required_columns(df):
        raise ValueError(
            f"Dataframe to be saved is missing (a) required column(s); has {df.columns}"
        )
    df.to_parquet(Path(path).with_suffix(".parquet"))
    with open(Path(path).with_suffix(".res"), "w") as stream:
        json.dump(resolution, stream)


def load(path) -> ty.Tuple[pd.DataFrame, ty.Dict]:
    df = pd.read_parquet(Path(path).with_suffix(".parquet"))
    if not _df_has_required_columns(df):
        raise ValueError(
            f"Loaded dataframe is missing (a) required column(s); has {df.columns}"
        )
    with open(Path(path).with_suffix(".res"), "r") as stream:
        resolution = json.load(stream)
    return df, resolution
