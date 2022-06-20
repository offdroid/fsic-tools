import os
import typing as ty

from fsic.convert import experiment_completed, file_paths, load_meta


def filter_experiments_in_dir(path, filter_fn: ty.Callable, full_path: bool = False):
    experiments = []
    for experiment_dir in os.listdir(path):
        meta_path, eval_path, _ = file_paths(
            os.path.join(path, experiment_dir), silent=True
        )
        if meta_path is None or eval_path is None:
            continue

        meta = load_meta(meta_path)
        if not experiment_completed(meta, silent=True):
            continue
        if filter_fn(meta):
            experiments.append(experiment_dir)

    if full_path:
        return [os.path.join(path, experiment) for experiment in experiments]
    else:
        return experiments
