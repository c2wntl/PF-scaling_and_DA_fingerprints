from src import loading_data, path_manipulation
import os,time
import subprocess
from itertools import product


if __name__ == '__main__':
    random_seed_id = 2
    config_names = ['test-to_public-configs'] # select the config files 
    # config_names = ['to_public-configs'] # select the config files 
    config_paths = [f"configs/{config_name}.yaml" for config_name in config_names]
    config = loading_data.merge_multiple_configs(config_paths) # merge the config files
    print(config)



    for database, model_name, target, dataset, feature_set in product(
        config['database'],
        config['model']['type'],
        config['target_column'],
        config['dataset'],
        config['feature_set'],
    ):
        root_path = os.path.dirname(os.path.abspath(__file__))
        
        save_path = path_manipulation.get_save_path(database,
                                                model_name,
                                                target,
                                                feature_set,
                                                dataset,random_seed_id=random_seed_id)

        log_file_path = os.path.join(save_path,'training.log')

        os.makedirs(save_path,exist_ok=True)

        command = f"nohup uv run python training_pipeline.py --database {database} --model_name {model_name} --target {target} --feature_set {feature_set} --dataset {dataset}  --random_seed_id {random_seed_id} >> {log_file_path} 2>&1 &"
        subprocess.run(command,shell=True, )
        time.sleep(1)