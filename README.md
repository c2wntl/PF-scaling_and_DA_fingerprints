# PF-Scaling and DA structural fingerprints

The alternative representation for complex defects in 2D materials.

This project is implemented in **Python >= 3.10.14**. Other dependencies are listed in the `pyproject.toml` file.

To reproduce our work, we recommend using [`uv`](https://docs.astral.sh/uv/), a fast Python package manager. Install it with:
```bash
$ pip install uv
```
After installing `uv` and cloning the main branch of this repository, run
```bash
$ uv sync
```
This will automatically generate a virtual python environment based on the dependencies specified in `pyproject.toml` and `uv.lock`  files.

The original CFID dataset is generated via `matminer` version `0.9.3`, and is stored in `./dataset/raw/` 

On the root directory there are four important scripts.

| file name                | task                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `feature_engineering.py` | Perform the PF-scaling and generate the DA fingerprints                                                                                                                                                                                                                                                                                                                                                   |
| `training_pipline.py`    | The main pipeline: loads the CFID descriptors, generates DA fingerprints, applies PF-scaling, optimizes models, and selects the best one.                                                                                                                                                                                                                                                                 |
| `main.py`                | Executes `training_pipeline.py` with user-specified inputs such as dataset name (`dataset`), feature set (`feature_set`), and target property (`target_column`). These inputs are defined in YAML config files in the `configs` directory. The random seed for train–test splits is controlled via `random_seed_list`. In this work, we use `random_seed_id = 2`, and results are stored in `results_2/`. |
| `run_tsne.py`            | Perform the t-SNE visualization. The results are saved in `results_2_tsne/` directory                                                                                                                                                                                                                                                                                                                     |

The general process of our experiments is creating a `YAML` config file, which identifies how to construct the extension of CFID by specify the `feature_set` (see the section below). Insert the name of such config file in `main.py`. Consequently, the training and model evaluation process can be started by 
```bash
$ uv run python main.py
```
After the finishing the training process, the t-SNE visualization can be implemented by
```bash
$ uv run python run_tsne.py
```
## Defining the `feature_set`

The operator used in PF-scaling, the distance type applied in DA structural fingerprints, and the inclusion or exclusion of distribution features are specified in the `feature_set` key of the config file, which is stored as a YAML file in `.\configs\`. For the original CFID, the definition is as follows:
```yaml
cfid # (Including distribution features)

chem_dist0_cfid # (Excluding distribution features)
```

### Apply only PF-Scaling
To perform PF-scaling, add the prefix 

```
vpa_{operator}_chem
``` 

to `cfid`, where `operator` can be 
- `subs` → subtraction
- `mult` → multiplication
- `divi` → division.
To remove all distribution features, add the tag `dist0`  after the above and before `cfid`.

For examples,

```yaml
# -- Including distribution features
- vpa_subs_chem_cfid        # (PF-Subtraction)
- vpa_mult_chem_cfid        # (PF-Multiplication)
- vpa_divi_chem_cfid        # (PF-Division)

# -- Excluding distribution features

- vpa_subs_chem_dist0_cfid  # (PF-Subtraction)
- vpa_mult_chem_dist0_cfid  # (PF-Multiplication)
- vpa_divi_chem_dist0_cfid  # (PF-Division)
```

### Apply only DA structural fingerprints
The prefix for adding the DA structural fingerprints to the feature set follows the 

```
{distance}_l1_pristine_
```

where `distance` can be:

- `tvd` → Total Variation Distance
- `hellinger` → Hellinger Distance
- `emd` → Earth Mover’s Distance

In contrast to PF-scaling, when including distribution features, append the `alldist` tag after `pristine`.
For examples,
```yaml
# -- Including distribution features

- tvd_l1_pristine_alldist_cfid          # (Total variation distance)
- hellinger_l1_pristine_alldist_cfid    # (Hellinger distance)
- emd_l1_pristine_alldist_cfid          # (Earth Mover distance)

# -- Excluding distribution features

- tvd_l1_pristine_cfid         # (Total variation distance)
- hellinger_l1_pristine_cfid   # (Hellinger distance)
- emd_l1_pristine_cfid         # (Earth Mover distance)
```
### Combining PF-Scaling and DA Structural Fingerprints

When combining both, place the PF-scaling tag before the DA fingerprint tag. The inclusion or exclusion of distribution features is again determined by the `alldist` tag. 

```
-- Including distribution features
{PF-scaling tag}_{DA fingerprint tag}_alldist_cfid

-- Excluding distribution features
{PF-scaling tag}_{DA fingerprint tag}_cfid
```

For examples, 

```yaml
# -- Including distribution features
- vpa_divi_chem_hellinger_l1_pristine_alldist_cfid   # (PF-division + Hellinger distance)
- vpa_subs_chem_emd_l1_pristine_alldist_cfid         # (PF-subtration + Earth Mover distance)

# -- Excluding distribution features
- vpa_divi_chem_hellinger_l1_pristine_cfid    # (PF-division + Hellinger distance)
- vpa_subs_chem_emd_l1_pristine_cfid          # (PF-subtration + Earth Mover distance)
```

## Training Pipeline and Results

In the `training_pipeline.py`, the `main_pipeline` function defines the complete workflow for model training and evaluation. It loads the CFID dataset, performs feature engineering (including PF-scaling or DA fingerprints, depending on the `feature_set`), splits the data into training and testing sets. The function then trains a default CatBoost model (via `base_model()` method), tunes hyperparameters using randomized search (`optimized`). The search results are saved in the `best_random/` directory, including the model with the lowest MAE. However, this model is not automatically considered the best one, as it only minimizes MAE without accounting for the train–test ratio. A very low MAE combined with a high train–test ratio may indicate overfitting. To address this, the `selected_optimized_model()` method is used to select the final model: it chooses the one with the lowest MAE while ensuring that the train–test MAE ratio does not exceed 1.5, and the selected model evaluation is saved in `selected_best_random_100`. The table below describes each result file.

- **`eval_results_from_cv.csv`** → Mean and standard deviation of evaluation metrics from 5-fold cross-validation.
- **`eval_metrics.csv`** → Evaluation metrics on the test set.
- **`feature_engineering_time.csv`** → Total time spent modifying the baseline CFID.
- **`feature_importance.csv`** → Feature importance scores.

> ⚠️ **Caution**  
> The cell-size–based descriptors generated with `matminer` version 0.9.3 are mislabeled (including the generated result files). 
> 
> The correct one should be as followed:  
> - `jml_pack_frac` (packing fraction)   →    `jml_log_vpa` (logarithm of VPA)  
> - `jml_vpa` (VPA)                                    →    `jml_pack_frac` (packing fraction)  
> - `jml_density` (density)                       →    `jml_vpa` (VPA)  
> - `jml_log_vpa` (logarithm of VPA)      →     `jml_density` (density)  
>
> Before analyzing the data please use `rename_cell_columns` method in `DataHandler.py` to correct them. 
> 
   This labeling issue also explains why feature sets involving PF-scaling are named with the prefix `vpa`.

---
## Preparing the Dataset

We provide the pre-generated baseline datasets for both defect structures from 2DMD, and the pristine structures. Please unzip and locate them explicitly at `./dataset/raw/.`
### Generating the Original CFID Dataset from Defect Structures

To generate the baseline CFID:
1. Download and unzip the dataset from 2DMD in `./dataset/database/

2. Run `gen_baseline_cfid.py` in `./dataset/database/` to compute the CFID for each host–defect density individually.
    
3. Each host–defect density CFID will be stored as a `.csv` file in `./dataset/database/tmp_file/`.
    
4. These temporary files are then merged using `merge_tmp_data.py` into:
    
    - `2dmd-cfid-all_high_density.csv` (high density)
        
    - `2dmd-cfid-all_low_density.csv` (low density)  
        Both located in `./dataset/raw/`.
### Generating the CFID Dataset from Pristine Structure

The pristine CFID, used for constructing the DA fingerprints, is generated from pristine structures in:  
`./dataset/raw/pristine_structure_cif_file/`  
using the script `gen_cfid_pristine_structure.py`.  
The final output is stored as:  
`./dataset/raw/2dmd-pristine_cfid-all_density.csv`

---

## Result Visualization

To visualize the simulated results displayed in the article, download and unzip:

- `results_2.zip`
    
- `results_2_tsne.zip`
    

to the root of this project, then explore the corresponding Jupyter notebooks in:  
`./notebooks/`

### Model Comparison

Due to incompatibilities between **`pycaret` v3.3.2** and **`pymatgen v2025.2.18`**, the model comparison of the baseline CFID across various models is demonstrated in:

- Notebook `0005`
    
- Notebook `0006`
    

in the `no-version-control` branch, which does not control the version of dependencies.

---



