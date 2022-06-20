import typing as ty

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

ACTIVE_COLOR = "orange"
PASSIVE_COLOR = "blue"
palette = {"active": ACTIVE_COLOR, "passive": PASSIVE_COLOR}


def _plot(**kwargs):
    if len(kwargs["data"]) == 0:
        return None
    ax = sns.lineplot(
        **kwargs,
    )

    xticker = ticker.EngFormatter(sep="", places=1)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: xticker.format_eng(
                (kwargs["data"].timesteps + 1).unique()[pos]
            )
        )
    )

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    return ax


def absolute(
    df: pd.DataFrame,
    x_axis: str = "length",
    separate_by: ty.Optional[ty.Dict[str, str]] = None,
):
    g = sns.FacetGrid(
        df.reset_index().replace({"active": {1.0: "active", 0.0: "passive"}}),
        **(separate_by or dict(col="buffer_size")),
    )
    g.map_dataframe(
        _plot,
        x="timesteps",
        y=x_axis,
        hue="active",
        palette=palette,
    )
    g.add_legend()


def percentage_optimal(
    df: pd.DataFrame,
    greater_than_eq_is_optimal: float,
    x_axis: str = "length",
    separate_by: ty.Optional[ty.Dict[str, str]] = None,
):
    separate_by = separate_by or dict(col="buffer_size", row="train_interval")
    combined_df = None
    for x in df[separate_by["col"]].unique():
        for y in df[separate_by["row"]].unique():
            _df = df[((df[separate_by["col"]] == x) &
                      (df[separate_by["row"]] == y))]
            for a in [True, False]:
                _adf = _df[_df["active"] == a]

                _bdf = (
                    _adf.groupby("timesteps")
                    .apply(lambda d: (d[x_axis].ge(greater_than_eq_is_optimal).sum()) / (d.shape[0]))
                    .to_frame("percentage_optimal")
                )
                for col in _adf.columns:
                    if col not in _bdf.columns and len(_adf[col].unique()) == 1:
                        _bdf[col] = _adf[col].iloc[0]
                combined_df = (
                    _bdf
                    if combined_df is None
                    else pd.concat([combined_df, _bdf.copy()])
                )
    assert combined_df is not None
    g = sns.FacetGrid(
        combined_df.reset_index().replace(
            {"active": {1.0: "active", 0.0: "passive"}}),
        **separate_by,
    )
    g.map_dataframe(
        _plot,
        x="timesteps",
        y="percentage_optimal",
        hue="active",
        palette=palette,
    )
    g.add_legend()
