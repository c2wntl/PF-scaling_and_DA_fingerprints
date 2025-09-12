import numpy as np
from src.data import read_csv_cif
import pandas as pd
from pathlib import Path
import os

def generate_weights(full_high_df=None,full_low_df=None):

    if (full_high_df is not None) and (full_low_df is not None):
        full_high_df.loc[:,'defect_density']  = 'high'
        full_high_df.loc[:,'N_part']  = 500
        full_low_df.loc[:,'defect_density']  = 'low'
        full_low_df.loc[:,'N_part']  = 5933

        full_df = pd.concat([full_high_df,full_low_df])

    elif (full_high_df is not None) and (full_low_df is None):
        full_high_df.loc[:,'defect_density']  = 'high'
        full_high_df.loc[:,'N_part']  = 500
        full_df = full_high_df
        
    elif (full_high_df is None) and (full_low_df is not None):
        full_low_df.loc[:,'defect_density']  = 'low'
        full_low_df.loc[:,'N_part']  = 5933
        full_df = full_low_df
    else:
        ValueError("One of full_high_df or full_low_df must be provided")


    full_df['base_density'] = full_df['base'] + "_" + full_df['defect_density']

    N_total = len(full_df) # total number of the data point
    C_parts = len(np.unique(full_df['base_density']))  # total number of host-defect_density, that is 8, (BN|P|InSe|GaSe|MoS2|WSe2)_high and (MoS2|WSe2)_low
    full_df['weight'] = N_total / (C_parts * full_df['N_part'])
    weights = full_df['weight']
    
    return full_df,weights,full_df['base_density']

def rename_cell_columns(df):
    rename_dict = {'jml_pack_frac':'jml_log_vpa',
                   'jml_vpa':'jml_pack_frac',
                   'jml_density':'jml_vpa',
                   'jml_log_vpa':'jml_density'}
    return df.rename(columns=rename_dict)

class OriginalLoad:
    def __init__(self,defect_density:str,host:str):
        project_path = os.path.dirname(os.path.realpath(f'{__file__}/../..'))
        dataset_dir =  f"{defect_density}_density_defects"
        self.example_dataset_path = Path(os.path.join(project_path,f"dataset/database/2d-materials-point-defects-all/{dataset_dir}/{host}"))
        self.sturctures = None
        self.defects = None
        self.targets = None

    def get_structure_defect(self):
        self.structures, self.defects = read_csv_cif(self.example_dataset_path)

    def get_targets(self):
        self.targets = pd.read_csv(self.example_dataset_path / "targets.csv.gz", index_col="_id")






    
    
class RawData:
    """
    Retrieve the data from the original database 
    """
    def __init__(self,structures:pd.DataFrame, defects:pd.DataFrame,targets:pd.DataFrame =None):
        self.structures = structures
        self.defects = defects
        self.reset_index_defects = defects.reset_index().rename(columns={'_id':'descriptor_id'})
        self.targets = targets
        
        self.merge_data = self._get_merge_data()

    def _get_merge_data(self):
        
        target_columns = self.targets.columns
        structure_columns = self.structures.columns

        intersect_columns = target_columns.intersection(structure_columns)
        
        structure_target_df = pd.merge(self.structures,self.targets.drop(columns=intersect_columns),how='inner',left_index=True,right_index=True)
        structure_target_defect_df =  pd.merge(structure_target_df,self.reset_index_defects, on = 'descriptor_id')
        return structure_target_defect_df.set_index(structure_target_df.index)

    def defect_descriptions(self):
        return self.merge_data['description'].value_counts()

        
    def filter_defect_type(self,defect_type):
        return self.merge_data[self.merge_data['description']==defect_type]
