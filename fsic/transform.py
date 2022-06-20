import typing as ty

import pandas as pd


def get_unique_indices(df: pd.DataFrame, index_id="uid") -> ty.Set[ty.Any]:
    return set(df[index_id].unique())


def add_column_from_map(
    df: pd.DataFrame, map: ty.Callable, new_column: str, index_id: str = "uid"
):
    index = get_unique_indices(df, index_id)
    index_map = {idx: map(idx) for idx in index}
    _df = df.copy()
    _df.insert(
        loc=0,
        column=new_column,
        value=df[index_id].map(index_map),
    )
    return _df


def get_experiment(resolution: ty.Dict, uid) -> ty.Dict:
    if uid not in resolution.keys():
        raise KeyError()
    v = resolution[uid]
    return v["meta"]["experiment"]


def add_active(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(df, lambda x: res[x]["training_pass"] == 0, "active")


def add_active_seed(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(df, lambda x: res[x]["meta"]["seed"], "active_seed")


def add_own_seed(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df, lambda x: res[x]["meta"]["seed"] +
        res[x]["training_pass"], "own_seed"
    )


def add_buffer_size(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["buffer_size"],
        "buffer_size",
    )


def add_batch_size(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["batch_size"],
        "batch_size",
    )


def add_train_interval(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["train_interval"],
        "train_interval",
    )


def add_learning_rate(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["learning_rate"],
        "learning_rate",
    )


def add_optimizer(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["optimizer"],
        "optimizer",
    )


def add_gradient_steps(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["gradient_steps"],
        "gradient_steps",
    )


def add_epsilon(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["exploration"][
            "Exponential"
        ]["init"],
        "epsilon",
    )


def augment_all(df: pd.DataFrame, resolution: ty.Dict) -> pd.DataFrame:
    fns = [
        add_active,
        add_active_seed,
        add_own_seed,
        add_buffer_size,
        add_batch_size,
        add_train_interval,
        add_learning_rate,
        add_optimizer,
        add_gradient_steps,
        add_epsilon,
        # Add additional column augmentation functions here
    ]
    for fn in fns:
        df = fn(df, resolution)
    return df


def aggregate(df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    return df.groupby(["uid", "pass_idx", "timesteps"])


def aggregate_mean(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df).mean()


def aggregate_median(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df).median()
