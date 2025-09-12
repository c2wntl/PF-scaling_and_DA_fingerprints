
from matminer.featurizers.structure.composite import JarvisCFID
import os
from pymatgen.core.structure import Structure
import pandas as pd



def get_pristine_2dmd_df():
    pristine_cif_file_dir = 'pristine_structure_cif_file'
    pristine_file_list = [f for f in os.listdir(pristine_cif_file_dir) if not f.startswith('.')]
    host_names = [h.split('_')[1][:-4] for h in pristine_file_list]
    cfid_pristine_X = []

    for pristine_file in pristine_file_list:
        print(pristine_file)
        pristine_structure = Structure.from_file(os.path.join(pristine_cif_file_dir,pristine_file))
        jarvis_object = JarvisCFID()
        cfid_pristine_X.append(jarvis_object.featurize(pristine_structure))
        
    cfid_pristine_df = pd.DataFrame(cfid_pristine_X,columns = jarvis_object.feature_labels())
    cfid_pristine_df['base'] = host_names

    return cfid_pristine_df
    
        
if __name__ == "__main__":

    pristine_df = get_pristine_2dmd_df()
    pristine_df.to_csv('2dmd-pristine_cfid-all_density.csv')