import hashlib
import logging
import os
import random
import typing as ty

import pandas as pd
import yaml


def load_meta(path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def experiment_completed(meta: ty.Dict, silent: bool = False):
    if meta.get("version", None) is None:
        if silent:
            return False
        else:
            raise RuntimeError(
                "Metadata file does not have version information")
    if meta["version"] < 2:
        if silent:
            return False
        else:
            raise RuntimeError("File format version is unsupported (< 2)")
    return meta.get("completed", False)


def file_paths(path, silent: bool = False):
    """Path is the parent for `meta.yaml`, `evals.parquet`, etc."""
    meta_path = os.path.join(path, "meta.yaml")
    meta_exists = os.path.isfile(meta_path)
    eval_path = os.path.join(path, "eval.parquet")
    eval_exists = os.path.isfile(meta_path)
    train_episode_lengths_path = os.path.join(
        path, "train_episode_lengths.parquet")
    train_episode_lengths_exists = os.path.isfile(meta_path)

    if not train_episode_lengths_exists and not silent:
        logging.warning("`train_episode_lengths.parquet` does not exist")
    if not meta_exists:
        if silent:
            meta_path = None
        else:
            raise RuntimeError("`meta.yaml` does not exist")
    if not eval_exists:
        if silent:
            eval_path = None
        else:
            raise RuntimeError("`eval.parquet` does not exist")
    return meta_path, eval_path, train_episode_lengths_path


def starts_with():
    pass


def load_raw_df(path, meta: ty.Dict, origin=None):
    df = pd.read_parquet(path)

    columns = df.columns
    columns = list(
        filter(
            lambda x: x.startswith("length")
            or x.startswith("reward")
            or x.startswith("failure"),
            columns,
        )
    )

    training_pass_indices = list(
        map(
            lambda x: int(x[len("length"):]),
            filter(
                lambda x: x.startswith("length"),
                columns,
            ),
        )
    )
    # Ensure all columns exist
    for idx in training_pass_indices:
        assert f"length{idx}" in df.columns
        assert f"reward{idx}" in df.columns
        assert f"failure{idx}" in df.columns

    if origin is None:
        origin = random.randint(0, int(2**32 - 1))
    elif not isinstance(origin, int):
        origin = int(hashlib.md5(repr(path).encode("utf-8")).hexdigest(), 16) % (
            2**32
        )
    assert isinstance(origin, int)

    merged_df = None
    resolution = {}
    for idx in training_pass_indices:
        subset = (
            df[["timesteps", f"length{idx}", f"reward{idx}", f"failure{idx}"]]
            .rename(
                columns={
                    f"length{idx}": "length",
                    f"reward{idx}": "reward",
                    f"failure{idx}": "failure",
                }
            )
            .copy()
        )
        subset["pass_idx"] = idx
        # Global unique id for every training pass
        uid = int(hashlib.md5(repr((origin, idx)).encode("utf-8")).hexdigest(), 16) % (
            2**32
        )
        subset["uid"] = uid

        if uid in resolution:
            raise RuntimeError(
                "All entries in resolution table must be unique, but id already exists"
            )
        resolution[uid] = {
            "meta": meta,
            "training_pass": idx,
        }

        merged_df = subset if merged_df is None else pd.concat([merged_df, subset])
    assert merged_df is not None
    # Unique id for every training. The same for all training passes of one file
    merged_df["origin"] = origin
    return (merged_df, resolution)


def convert(path):
    meta_path, eval_path, _ = file_paths(path)
    meta = load_meta(meta_path)
    assert experiment_completed(meta)

    return load_raw_df(eval_path, meta)


def convert_iter(iterator) -> ty.Iterator:
    for path in iterator:
        yield convert(path)
