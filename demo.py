import matplotlib.pyplot as plt

from fsic import convert, io, merge, plot, query, transform

PATH_TO_EXPERIMENTS = "/path/to/experiments"

EXPERIMENT_PREFIX = "some_prefix"
# Query / filter experiments by some callback.
# Here we filter all experiment which have a (experiment) name starting with a given prefix.
experiment_paths = query.filter_experiments_in_dir(
    PATH_TO_EXPERIMENTS,
    lambda x: x["experiment"]["name"].startswith(EXPERIMENT_PREFIX),
    True,
)

# Convert raw dataframes into pairs of dataframes (more flexible) and resolution tables.
# The resolution table contains information, like meta-data, on the experiments.
# Later we will add this information, e.g., buffer capacity, as a column to the dataframe.
converted = convert.convert_iter(experiment_paths)

# Merge all selected experiments into a single dataframe
df, res = merge.merge_iter(converted)

# Add information from the resolution table (the contained meta-data) as rows to the dataframe.
# `transform.augment_all()` is a helper that does this for all supported fields.
# Needs to be updated when other fields are added!
df = transform.augment_all(df, res)

# Aggregate over all trials for every trainings (uid) at each timestep, while respecting the active/passive value.
# This does not aggregate all trainings!
df = transform.aggregate_mean(df)

# Save and load the dataframe & resolution table, it will be saved as two files, `.parquet` and `.res`, respectively.
EXPORT_PATH = "./demo_data"
io.save(df, res, EXPORT_PATH)
del df, res  # Delete the variables
df, res = io.load(EXPORT_PATH)

# Plot the final dataframe in absolute form
plot.absolute(
    df, x_axis="length", separate_by=dict(col="buffer_size", row="train_interval")
)
# and as a percentage
plot.percentage_optimal(
    df,
    3000.0,
    x_axis="length",
    separate_by=dict(col="buffer_size", row="train_interval"),
)
plt.show()
