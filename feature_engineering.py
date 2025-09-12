from src import training_predictor
from src import loading_data, path_manipulation, model_evaluation
from src.loading_data import split_cfid
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os, sys
from src import distances
import re
import pandas as pd
import numpy as np


class DataPreprocessing:
    def __init__(self):
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        with open(
            os.path.abspath(os.path.join(self.root_dir, "feature_sets/cfid.json")), "r"
        ) as file:
            self.cfid_features = json.load(file)

    def filtered_prefix_cfid(self, df: pd.DataFrame, prefix: str):
        features = [
            f for f in self.cfid_features["num_feat"] if f.startswith("jml_" + prefix)
        ]
        return df[features]

    def filtered_non_prefix_cfid(self, df: pd.DataFrame, prefix_list: list):
        non_prefix_features = [
            f
            for f in self.cfid_features["num_feat"]
            if not any(f.startswith("jml_" + prefix) for prefix in prefix_list)
        ]
        return df[non_prefix_features]

    def normalizing_data(self, X: pd.DataFrame, scaler: str):
        if scaler is None:
            transformed_X = X
        elif scaler == "l1":
            transformed_X = X.div(X.sum(axis=1), axis=0)
        elif scaler == "l2":
            l2_norm = np.sqrt((X**2).sum(axis=1))
            transformed_X = X.div(l2_norm, axis=0)
        elif scaler == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            mm_scaler = MinMaxScaler()
            mm_scaler.fit(X)
            transformed_X = mm_scaler.transform(X)
            transformed_X = pd.DataFrame(
                data=transformed_X, columns=X.columns, index=X.index
            )
        else:
            scaler.fit(X)
            transformed_X = scaler.transform(X)
            transformed_X = pd.DataFrame(
                data=transformed_X, columns=X.columns, index=X.index
            )

        return transformed_X

    def calculate_distance(
        self, X: pd.DataFrame, distance: str, prefix: str, x_ref=None
    ): 
        if x_ref is None:
            x_ref = pd.DataFrame([np.zeros(len(X.columns))], columns=X.columns)
        distance_obj = getattr(distances, distance)

        return distance_obj(X, x_ref.iloc[0]).rename(f"{prefix}_{distance}")

    def stratified_split_train_test_data(
        self, X, y, test_size, random_state, base_density
    ):
        """
        Perform a stratified train-test split based on target bins and a density variable.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target variable.
            base_density (pd.Series): Density variable used for stratification.

        Returns:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.Series): Training target values.
            y_test (pd.Series): Testing target values.
        """
        target = y.name
        # Merge features, target, and base_density into one DataFrame for stratification
        df = pd.merge(X, y, left_index=True, right_index=True)
        df = pd.merge(df, base_density, left_index=True, right_index=True)

        # Define minimum samples per bin and the initial number of bins
        mins_samples_per_bin = 2
        num_bins = 10

        # Dynamically adjust bins to ensure all bins have the required minimum samples
        while True:
            # Create bins for the target variable
            bins = np.linspace(y.min() - 0.1, y.max(), num_bins)
            df.loc[:, f"{target}_bin"] = pd.cut(df[target], bins=bins, labels=False)

            # Count the number of samples in each stratification group (target_bin, base_density)
            bin_counts = df[[f"{target}_bin", "base_density"]].value_counts().values

            # Check if all bins meet the minimum sample requirement
            if all(bin_counts >= mins_samples_per_bin):
                break
            else:
                # Reduce the number of bins if the condition is not satisfied
                num_bins -= 1
                if num_bins < 2:
                    raise ValueError(
                        "Could not create a binning scheme with at least two samples per bin."
                    )

        # Perform stratified shuffle split using the bin and base_density columns
        split = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        for train_index, test_index in split.split(
            df, df[[f"{target}_bin", "base_density"]]
        ):
            train_set = df.iloc[train_index]
            test_set = df.iloc[test_index]

        # Drop helper columns before returning the splits
        train_set = train_set.drop(columns=[f"{target}_bin", "base_density"])
        test_set = test_set.drop(columns=[f"{target}_bin", "base_density"])

        # Separate features and target variables for the final output
        X_train, y_train = train_set.drop(target, axis=1), train_set[target]
        X_test, y_test = test_set.drop(target, axis=1), test_set[target]

        return X_train, X_test, y_train, y_test


class DimensionalReduction:
    """
    This is the main class for performing the PF-scaling, and construct DA structural fingerprints.
    """
    def __init__(
        self,
        X,
        y,
        how: str = None,
        bases: pd.Series = None,
        standard_scaler=True,
        base_density=None,
        random_seed_id=None,
    ):
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        # split_random_state = 52
        if random_seed_id is not None:
            random_state_dict = {
                "0": 2,
                "1": 12,
                "2": 22,
                "3": 32,
                "4": 42,
                "5": 52,
                "6": 62,
                "7": 72,
                "8": 82,
                "9": 92,
                "10": 102,
                "11": 112,
                "12": 122,
                "13": 132,
                "14": 142,
                "15": 152,
                "16": 162,
                "17": 172,
                "18": 182,
                "19": 192,
                "20": 202,
                "21": 212,
                "22": 222,
                "23": 232,
                "24": 324,
                "25": 325,
                "26": 326,
                "27": 327,
                "28": 328,
                "29": 329,
                "30": 330,
            }
            split_random_state = random_state_dict[str(random_seed_id)]
        else:
            split_random_state = 42
        self.split_random_state = split_random_state
        test_size = 0.2

        is_vpa = False
        is_cal_distance = False
        is_pristine_ref = False
        is_pca = False
        is_alldist = False

        if "vpa" in how:
            is_vpa = True

        if ("hellinger" in how) or ("jsd" in how) or ("emd" in how) or ("tvd" in how):
            is_cal_distance = True

        if "pristine" in how:
            is_pristine_ref = True

        if "alldist" in how:
            is_alldist = True

        if "pca" in how:
            is_pca = True
            pattern = r"pca_cfid_(\d+)"
            match = re.search(pattern, how)
            if match:
                n_comp_pca = int(match.group(1))
            else:
                raise ValueError("Invalid PCA pattern, it should be pca_cfid_<n_comp>")

        ################ other reduction techniques ################

        if is_pca:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                DataPreprocessing().stratified_split_train_test_data(
                    X,
                    y,
                    test_size=test_size,
                    random_state=split_random_state,
                    base_density=base_density,
                )
            )
            (
                self.X_train_transform,
                self.X_test_transform,
                self.y_train_transform,
                self.y_test_transform,
                self.features,
            ) = self.get_pca(n_comp_pca)
        elif how == "kazeev":
            self.X_train, self.X_test, self.y_train, self.y_test = (
                DataPreprocessing().stratified_split_train_test_data(
                    X,
                    y,
                    test_size=test_size,
                    random_state=split_random_state,
                    base_density=base_density,
                )
            )
            self.X_train_transform, self.X_test_transform = self.apply_standard_scaler(
                self.X_train, self.X_test
            )
            self.y_train_transform = self.y_train
            self.y_test_transform = self.y_test
            self.features = X.columns
            ################ chemical part ###################
        else:
            if is_vpa:
                pattern = r"vpa_(divi|mult|subs)_(eo_chem|chem)"
                match = re.search(pattern, how)
                if match:
                    operator = match.group(1)
                    chem_part = match.group(2)

                    vpa_chem_part_df = self.get_vpa_operation(X, operator, chem_part)
                    chem_part_df = pd.merge(
                        vpa_chem_part_df,
                        split_cfid(X, is_cell=True),
                        left_index=True,
                        right_index=True,
                    )
                else:
                    raise ValueError(f"Invalid pattern, it should be {pattern}")

            else:
                # in case there is no engineering in chemical part, it will filter the dataframe only to be eo_chem or chem
                if ("eo" in how) & ("chem" in how):
                    chem_part_df = split_cfid(X, is_eo_chem=True, is_cell=True)
                elif ("eo" not in how) & ("chem" in how):
                    chem_part_df = split_cfid(X, is_chem=True, is_cell=True)
                else:
                    chem_part_df = split_cfid(X, is_chem=True, is_cell=True)

            ############### distribution part  ###############
            if is_cal_distance:
                pattern = r"(hellinger|emd|tvd|jsd)_(l1|l2)"
                match = re.search(pattern, how)
                if match:
                    distance = match.group(1)
                    normalization = match.group(2)
                    print(f"{distance} - {normalization}")
                    if is_pristine_ref:
                        X_ig, _ = self.pristine_ig(
                            X,
                            distance,
                            normalization,
                            prefix_list=["rdf", "adf1", "adf2", "ddf", "nn"],
                            bases=bases,
                        )
                    else:
                        X_ig, _ = self.absolute_ig(
                            X,
                            distance,
                            normalization,
                            prefix_list=["rdf", "adf1", "adf2", "ddf", "nn"],
                        )
                    X_distance = X_ig.loc[:, X_ig.columns.str.contains(distance)]
                else:
                    raise ValueError(f"Invalid pattern, it should be {pattern}")

                if is_alldist:
                    X_alldist = split_cfid(X, is_distribution=True, is_chrg=True)
                    dist_part_df = pd.merge(
                        X_alldist, X_distance, left_index=True, right_index=True
                    )
                else:
                    X_chrg = split_cfid(X, is_chrg=True)
                    dist_part_df = pd.merge(
                        X_distance, X_chrg, left_index=True, right_index=True
                    )

            else:
                pattern = r"_(dist(\d+))_"
                match = re.search(pattern, how)
                if match:
                    distn_dict = {
                        "dist0": [" "],
                        "dist1": ["nn"],
                        "dist2": ["nn", "ddf", "adf1"],
                    }
                    distn = match.group(1)
                    distn_cfid = self.filterd_df_by_column_names(X, distn_dict[distn])
                    X_chrg = split_cfid(
                        X, is_chrg=True
                    )  
                    dist_part_df = pd.merge(
                        distn_cfid, X_chrg, left_index=True, right_index=True
                    )  
                else:
                    dist_part_df = split_cfid(X, is_distribution=True, is_chrg=True)

            engineered_X = pd.merge(
                chem_part_df, dist_part_df, left_index=True, right_index=True
            )

            self.X_train, self.X_test, self.y_train, self.y_test = (
                DataPreprocessing().stratified_split_train_test_data(
                    X=engineered_X,
                    y=y,
                    test_size=test_size,
                    random_state=split_random_state,
                    base_density=base_density,
                )
            )

            if standard_scaler:
                self.X_train_transform, self.X_test_transform = (
                    self.apply_standard_scaler(self.X_train, self.X_test)
                )
                self.y_train_transform = self.y_train
                self.y_test_transform = self.y_test
                self.features = engineered_X.columns
            else:
                self.X_train_transform = self.X_train
                self.X_test_transform = self.X_test
                self.y_train_transform = self.y_train
                self.y_test_transform = self.y_test
                self.features = engineered_X.columns

    def filterd_df_by_column_names(self, df, column_names: list):
        return df.loc[:, df.columns.str.contains("|".join(column_names))]

    def apply_standard_scaler(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_transform = scaler.fit_transform(X_train)
        X_test_transform = scaler.transform(X_test)

        return X_train_transform, X_test_transform

    def get_vpa_operation(self, X, operator, chem_part):
        if chem_part == "chem":
            chem_part_df = split_cfid(X, is_chem=True)
        elif chem_part == "eo_chem":
            chem_part_df = split_cfid(X, is_eo_chem=True)

        cell_df = split_cfid(X, is_cell=True)
        vpa_df = cell_df["jml_vpa"]
        vpa_opt_chem_part = self.calculate(chem_part_df, vpa_df, operator)
        return vpa_opt_chem_part

    def calculate(self, X, x, operator: str):
        operations = {
            "divi": X.div,
            "mult": X.mul,
            "subs": X.sub,
        }

        if operator in operations:
            x_opt_X = operations[operator](x, axis=0)
            return x_opt_X
        else:
            raise ValueError(
                f"Invalid operator '{operator}'. Choose from 'divi', 'mult', 'subs'."
            )

    def get_pca(self, n_components):
        scaler = StandardScaler()
        X_train_transform = scaler.fit_transform(self.X_train)
        X_test_transform = scaler.transform(self.X_test)

        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_transform)
        X_test_pca = pca.transform(X_test_transform)

        features = [f"pca_{i}" for i in np.arange(n_components)]

        return X_train_pca, X_test_pca, self.y_train, self.y_test, features

    def absolute_ig(self, X, distance, normalization, prefix_list):
        dp = DataPreprocessing()

        X_non_prefices = dp.filtered_non_prefix_cfid(X, prefix_list)

        X_distance_prefices = pd.DataFrame(index=X.index)  # initialize df
        normalized_X_prefices = pd.DataFrame(index=X.index)  # initialize df

        for prefix in prefix_list:
            # filtering the features
            X_prefix = dp.filtered_prefix_cfid(X, prefix)

            # normalizing the features
            normalized_X_prefix = dp.normalizing_data(X_prefix, scaler=normalization)

            # merging normalizing data
            normalized_X_prefices = pd.merge(
                normalized_X_prefices,
                normalized_X_prefix,
                left_index=True,
                right_index=True,
            )

            # calculating the distance
            distance_X_prefix = dp.calculate_distance(
                normalized_X_prefix, distance=distance, prefix=prefix
            )

            # merging distance prefices
            X_distance_prefices = pd.merge(
                X_distance_prefices,
                distance_X_prefix,
                left_index=True,
                right_index=True,
            )

        X_ig = pd.merge(
            X_distance_prefices, X_non_prefices, left_index=True, right_index=True
        )

        return X_ig, normalized_X_prefices



    def pristine_ig(self, X, distance, normalization, prefix_list, bases):
        dp = DataPreprocessing()

        X_non_prefices = dp.filtered_non_prefix_cfid(X, prefix_list)

        X_distance_prefices = pd.DataFrame(index=X.index)  # initialize df
        normalized_X_prefices = pd.DataFrame(index=X.index)  # initialize df
        filtered_bases = bases.loc[X.index]

        pristine_cfid = pd.read_csv(
            os.path.join(
                self.root_dir, "dataset/raw/2dmd-pristine_cfid-all_density.csv"
            ),
            index_col=0,
        )

        for prefix in prefix_list:
            # filtering the features
            X_prefix = dp.filtered_prefix_cfid(X, prefix)

            # normalizing the features
            normalized_X_prefix = dp.normalizing_data(X_prefix, scaler=normalization)

            # merging normalized data
            normalized_X_prefices = pd.merge(
                normalized_X_prefices,
                normalized_X_prefix,
                left_index=True,
                right_index=True,
            )

            # calculating the distance
            distance_X_prefix = pd.DataFrame()
            for host in np.unique(filtered_bases):
                x_ref = pristine_cfid[pristine_cfid["base"] == host].drop(
                    columns=["base"]
                )
                x_ref = dp.filtered_prefix_cfid(x_ref, prefix)
                x_ref = dp.normalizing_data(x_ref, normalization)
                host_bases_index = filtered_bases[filtered_bases == host].index
                host_normalized_X_prefix = normalized_X_prefix.loc[
                    host_bases_index, :
                ].copy()
                host_distance_X_prefix = dp.calculate_distance(
                    host_normalized_X_prefix, distance, prefix, x_ref
                )
                distance_X_prefix = pd.concat(
                    [distance_X_prefix, host_distance_X_prefix]
                )

            # merging distance prefices
            X_distance_prefices = pd.merge(
                X_distance_prefices,
                distance_X_prefix,
                left_index=True,
                right_index=True,
            )

        X_ig = pd.merge(
            X_distance_prefices, X_non_prefices, left_index=True, right_index=True
        )

        return X_ig, normalized_X_prefices

    def get_ig(
        self,
        distance: str,
        normalization,
        pristine=False,
        bases=None,
        standard_scaler=True,
    ):
        prefix_list = ["mean_charge", "rdf", "adf1", "adf2", "ddf", "nn"]

        if pristine:
            X_train_ig, normalized_X_train_prefices = self.pristine_ig(
                self.X_train, distance, normalization, prefix_list, bases
            )
            X_test_ig, normalized_X_test_prefices = self.pristine_ig(
                self.X_test, distance, normalization, prefix_list, bases
            )
        else:
            X_train_ig, normalized_X_train_prefices = self.absolute_ig(
                self.X_train, distance, normalization, prefix_list
            )
            X_test_ig, normalized_X_test_prefices = self.absolute_ig(
                self.X_test, distance, normalization, prefix_list
            )

        self.X_train_ig = X_train_ig
        self.X_test_ig = X_test_ig

        self.normalized_X_train_prefices = normalized_X_train_prefices
        self.normalized_X_test_prefices = normalized_X_test_prefices

        features = X_train_ig.columns
        if standard_scaler:
            scaler = StandardScaler()
            X_train_ig_transform = scaler.fit_transform(X_train_ig)
            X_test_ig_transform = scaler.transform(X_test_ig)
        else:
            X_train_ig_transform = X_train_ig
            X_test_ig_transform = X_test_ig

        return (
            X_train_ig_transform,
            X_test_ig_transform,
            self.y_train,
            self.y_test,
            features,
        )
