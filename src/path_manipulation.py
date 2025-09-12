import os
from pathlib import Path

def get_save_path(database,model_name,target,feature_set,dataset,random_seed_id=None,optimize=None):
    if random_seed_id is not None:
        result_dir = f'results_{random_seed_id}'
    else:
        result_dir = 'result'        
    
    if optimize is not None:
        save_path = os.path.join(result_dir,database,model_name,target,feature_set,dataset,optimize)
    else:
        save_path = os.path.join(result_dir,database,model_name,target,feature_set,dataset)
    return save_path
