import pandas as pd
import os, json
import yaml
from joblib import load
from src.utils import DataHandler

src_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(src_dir, ".."))


def full_data(database, dataset, feature_set):
    return pd.read_csv(
        f"{project_path}/dataset/raw/{database}-{feature_set}-{dataset}.csv",
        index_col=0,
    )


def get_full_all_density_data():
    all_high_density = full_data("2dmd", "all_high_density", "cfid")
    all_high_density["density"] = "high"
    all_low_density = full_data("2dmd", "all_low_density", "cfid")
    all_low_density["density"] = "low"
    all_density = pd.concat([all_high_density, all_low_density], axis=0)
    all_density = DataHandler.rename_cell_columns(all_density)

    return all_density


def split_X_y(full_data_df: pd.DataFrame, feature_set: str, target: str):
    with open(f"{project_path}/feature_sets/{feature_set}.json") as file:
        json_file = json.load(file)
    features = json_file["num_feat"] + json_file["char_feat"]
    X = full_data_df[features]
    y = full_data_df[target]
    return X, y


def select_host_X_y(X, y, bases, host):
    host_indices = bases[bases.values == host].index
    host_X = X.loc[host_indices, :]
    host_y = y.loc[host_indices]
    return host_X, host_y


def merge_configs(config1, config2):
    # Initialize the resulting dictionary
    C = {}

    for key in config1:
        if isinstance(config1[key], list) and isinstance(config2[key], list):
            # Merge lists and remove duplicates
            C[key] = list(set(config1[key] + config2[key]))
        elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
            # Recursively merge nested dictionaries
            C[key] = merge_configs(config1[key], config2[key])
        else:
            # If the key is not a list or dict, just take the value from A (assuming A and B have the same non-list, non-dict values for the same keys)
            C[key] = config1[key]

    return C


def load_config_file(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_pristine_df():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(src_dir, ".."))
    return pd.read_csv(
        os.path.join(root_dir, "dataset/raw/2dmd-pristine_cfid-all_density.csv"),
        index_col=0,
    )


def merge_multiple_configs(config_paths: list):
    merged_config = {
        "database": [],
        "feature_set": [],
        "dataset": [],
        "target_column": [],
        "model": {"type": []},
    }

    # Load and merge configs one by one
    for config_path in config_paths:
        config = load_config_file(config_path)
        merged_config = merge_configs(merged_config, config)

    return merged_config


def load_results(
    filename,
    database,
    feature_set,
    dataset,
    target_column,
    model,
    optimize=None,
    result_dirname="results",
    abs_save_path=None,
):
    project_path = os.path.dirname(os.path.realpath(f"{__file__}/.."))
    if abs_save_path is None:
        results_path = os.path.join(
            project_path,
            result_dirname,
            database,
            model,
            target_column,
            feature_set,
            dataset,
        )
        if optimize is None:
            save_path = os.path.join(results_path, filename)
        elif optimize == "best_random":
            save_path = os.path.join(results_path, optimize, filename)
        elif optimize == "gross_grid":
            save_path = os.path.join(results_path, optimize, filename)
        elif optimize == "fine_grid":
            save_path = os.path.join(results_path, optimize, filename)
        else:
            save_path = os.path.join(results_path, optimize, filename)
    else:
        save_path = abs_save_path

    file_type = filename.split(".")[-1]

    if file_type == "csv":
        results = pd.read_csv(save_path, index_col=0)
    elif file_type == "pkl":
        results = load(save_path)
    elif file_type == "json":
        with open(save_path, "r") as file:
            results = json.load(file)
    elif file_type == "txt":
        pass

    return results


def split_cfid(
    X: pd.DataFrame,
    distance=None,
    is_chem=False,
    is_eo_chem=False,
    is_cell=False,
    is_distribution=False,
    is_chrg=False,
    is_rdf=False,
    is_adf1=False,
    is_adf2=False,
    is_ddf=False,
    is_nn=False,
):
    features = []
    srcdir = os.path.dirname(os.path.realpath(__file__))
    rootdir = os.path.abspath(os.path.join(srcdir, ".."))
    feature_sets_dir = os.path.join(rootdir, "feature_sets")

    with open(os.path.join(feature_sets_dir, "chem_cfid.json"), "r") as file:
        chem_cfid = json.load(file)["num_feat"]
    with open(os.path.join(feature_sets_dir, "cfid.json"), "r") as file:
        cfid = json.load(file)["num_feat"]

    if is_chem and is_eo_chem is False:
        features += chem_cfid
    elif is_eo_chem and is_chem is False:
        chem_eo_cfid = [
            f
            for f in chem_cfid
            if not any(sub in f for sub in ["mult", "divi", "add", "subs"])
        ]
        features += chem_eo_cfid
    elif is_eo_chem and is_chem:
        ValueError(f"is_chem and is_eo_chem cannot be both True")

    if is_chrg:
        chrg_cfid = [f for f in cfid if "mean_charge" in f]
        features += chrg_cfid

    if is_cell:
        cell_cfid = ["jml_pack_frac", "jml_vpa", "jml_density", "jml_log_vpa"]
        features += cell_cfid

    if distance is not None:
        features += [f"{f}_{distance}" for f in ["rdf", "nn", "ddf", "adf1", "adf2"]]

    if is_distribution:
        labels = []
        labels += [f"jml_rdf_{i}" for i in range(1, 101)]
        for lvl in [1, 2]:
            labels += [f"jml_adf{lvl}_{i}" for i in range(1, 180)]
        labels += [f"jml_ddf_{i}" for i in range(1, 180)]
        labels += [f"jml_nn_{i}" for i in range(1, 101)]
        features += labels
    elif is_rdf:
        # features=[]
        features += [f"jml_rdf_{i}" for i in range(1, 101)]
    elif is_nn:
        # =[]
        features += [f"jml_nn_{i}" for i in range(1, 101)]
    elif is_ddf:
        # labels=[]
        features += [f"jml_ddf_{i}" for i in range(1, 180)]
    elif is_adf1:
        # labels=[]
        features += [f"jml_adf1_{i}" for i in range(1, 180)]
    elif is_adf2:
        # labels=[]
        features += [f"jml_adf2_{i}" for i in range(1, 180)]

    return X.loc[:, features]


def filter_config_df(df, database, model, target_column, feature_set, dataset):
    return df[
        (df["database"] == database)
        & (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["target_column"] == target_column)
        & (df["feature_set"] == feature_set)
    ]
