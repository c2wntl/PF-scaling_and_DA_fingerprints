import pandas as pd 
from matminer.featurizers.structure.composite import JarvisCFID
from data import read_csv_cif
from pathlib import Path
from tqdm import tqdm
import os

def get_data_from_2dmd(dir_name:str):
    db_dir = '2d-materials-point-defects-all'
    db_path = Path(os.path.join(db_dir,dir_name))


    structures, defects = read_csv_cif(db_path)
    targets = pd.read_csv(db_path / "targets.csv.gz", index_col="_id")

    flat = structures.combine_first(targets).merge(defects, how="left", left_on="descriptor_id", right_index=True)

    return flat

def convert_dict_to_df(data:list):
    """
    This is the helper function of gen_host_density_cfid_baseline()
    """
    tmp_df = pd.DataFrame(data)
    tmp_df = tmp_df.set_index(tmp_df['index'])
    tmp_df.index.name = None
    tmp_df.drop(columns=['index'],inplace=True)
    return tmp_df


def gen_host_density_cfid_baseline(source_df:pd.DataFrame):
    data_dict = []
    k=0
    for i,row in tqdm(source_df.iterrows()):
        features_dict = {}
        featurizer = JarvisCFID()
        s = row.initial_structure
        features = featurizer.featurize(s)
        features_dict = {key:val for key,val in zip(featurizer.feature_labels(),features)}
        features_dict['index'] = i
        
        data_dict.append(features_dict)

        ########## remove this 
        k+= 1
        if k==20:
            break

    data_df = convert_dict_to_df(data_dict)


    return data_df

def split_label(filename:str):
    label_list = filename.split('/')
    density = label_list[0]
    host = label_list[-1]
    return density, host

if __name__ == "__main__":
    filenames = {
                        # 'high_density_defects/BP_spin_500':'BP',
                        # 'high_density_defects/GaSe_spin_500':'GaSe',
                        # 'high_density_defects/InSe_spin_500':'InSe',
                        'high_density_defects/MoS2_500':'MoS2',
                        # 'high_density_defects/WSe2_500':'WSe2',
                        # 'high_density_defects/hBN_spin_500':'BN',
                        # 'low_density_defects/MoS2':'MoS2',
                        # 'low_density_defects/WSe2':'WSe2'
                        
    }

    # print('hello')
    for filename,host in filenames.items(): 
        print(filename,host)
        source_data_df = get_data_from_2dmd(filename)
        cfid_df = gen_host_density_cfid_baseline(source_data_df)
        
        density,_ = split_label(filename)
        cfid_df = pd.merge(cfid_df,source_data_df,left_index=True,right_index=True)
        cfid_df.to_csv(f'tmp_data/tmp-2dmd-cfid-{host}_{density}.csv')
