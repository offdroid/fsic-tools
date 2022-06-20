import copy
import typing as ty

import pandas as pd


def _df_has_required_columns(df) -> bool:
    # Df must not be indexed by any of the following columns
    return all(
        map(
            lambda x: x in df.columns,
            ["timesteps", "length", "reward", "failure", "origin", "uid"],
        )
    )


def merge(df_1, resolution_1, df_2, resolution_2) -> ty.Tuple:
    res_1_keys = set(resolution_1.keys())
    res_2_keys = set(resolution_2.keys())
    if not res_1_keys.isdisjoint(res_2_keys):
        raise RuntimeError(
            f"Resolution tables must be disjoint, but {res_1_keys.intersection(res_2_keys)} are found in both"
        )
    combined_resolution = resolution_1 | resolution_2
    if not _df_has_required_columns(df_1):
        raise ValueError("Dataframe 1 is missing (a) required column(s)")
    if not _df_has_required_columns(df_2):
        raise ValueError("Dataframe 2 is missing (a) required column(s)")
    combined_df = pd.concat([df_1, df_2])

    return combined_df, combined_resolution


def merge_iter(iterator: ty.Iterator) -> ty.Tuple:
    combined_resolution = None
    combined_df = None
    for idx, (df, res) in enumerate(iterator):
        if combined_resolution is None:
            combined_resolution = copy.deepcopy(res)
            if not _df_has_required_columns(df):
                raise ValueError("Dataframe 1 is missing (a) required column(s)")
            combined_df = df
        else:
            assert combined_resolution is not None
            res_1_keys = set(combined_resolution.keys())
            res_2_keys = set(res.keys())
            if not res_1_keys.isdisjoint(res_2_keys):
                raise RuntimeError(
                    f"Resolution tables must be disjoint, but {res_1_keys.intersection(res_2_keys)} are found two dataframes (indx = {idx})"
                )
            combined_resolution = combined_resolution | res
            if not _df_has_required_columns(df):
                raise ValueError(f"Dataframe {idx} is missing (a) required column(s)")
            assert combined_df is not None
            combined_df = pd.concat([combined_df, df])
    return combined_df, combined_resolution
