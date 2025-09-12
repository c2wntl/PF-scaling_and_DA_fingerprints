from src import loading_data,training_predictor,path_manipulation
from src.utils import plot,DataHandler,ResultsHandler
from sklearn.model_selection import KFold,learning_curve
from feature_engineering import DataPreprocessing

import pandas as pd
import os,json
import numpy as np

from itertools import product
from matplotlib import pyplot as plt
import training_pipeline


class LearningCurve():
    
    def __init__(self,config,random_seed_id,optimization):
        self.config = config
        self.random_seed_id = random_seed_id
        self.optimization = optimization
        if random_seed_id is not None:
            self.result_dir = f'results_{random_seed_id}'
        else:
            self.result_dir = 'results'
        
        self.model = config['model']
        self.target_column = config['target_column']
        self.dataset = config['dataset']
        self.database = config['database']
        self.feature_set = config['feature_set']
        
        save_path = path_manipulation.get_save_path(database = self.database,
                                                         model_name = self.model,
                                                         target = self.target_column,
                                                         feature_set = self.feature_set,
                                                         dataset = self.dataset,
                                                         random_seed_id=random_seed_id)
        
        self.save_path = os.path.join(save_path,optimization,'learning_curve')
        os.makedirs(self.save_path,exist_ok=True)
        
    def load_train_Xy(self):
        X_train = loading_data.load_results('X_train.csv', result_dirname=self.result_dir, **self.config)       
        y_train = loading_data.load_results('y_train.csv', result_dirname=self.result_dir, **self.config)       
        return X_train,y_train
    
    def load_sample_weight(self):
        return loading_data.load_results('sample_weight.csv',result_dirname=self.result_dir, **self.config)

    def load_optimize_params(self):
        if self.optimization == 'best_random' or self.optimization == 'gross_grid':
            opt_params = loading_data.load_results('best_params.json',optimize=self.optimization,result_dirname=self.result_dir, **self.config)
        else:
            meta_data = loading_data.load_results('meta_data.json',optimize=self.optimization,result_dirname=self.result_dir, **self.config)
            opt_params = meta_data['optimized_params']
        return opt_params

    def compute_learning_curve(self):

        X_train,y_train = self.load_train_Xy()
        y_train = y_train.iloc[:,0]
        sample_weight = self.load_sample_weight()
        params = self.load_optimize_params()



        training_model = training_predictor.initialize_model(model_name=self.model,params=params,
                                                            n_jobs=None)

        scoring = {'weighted_mae':training_predictor.weighted_mae_scorer(sample_weight['weight'])}
        kfold = KFold(n_splits=5,shuffle=True,random_state=42)

        train_sizes, train_scores, test_scores = learning_curve(
                                                                training_model,X_train,y_train, cv=kfold, scoring=scoring['weighted_mae'],
                                                                random_state=42,shuffle=True,
                                                                train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1,verbose=3
                                                                )
        
        learning_curve_data = {
            "train_sizes": train_sizes.tolist(),
            "train_scores": train_scores.tolist(),
            "test_scores": test_scores.tolist()
        }
        
        with open(os.path.join(self.save_path,"learning_cruve_results.json"),"w") as f:
            json.dump(learning_curve_data, f, indent= 4)

        return train_sizes, train_scores, test_scores



def generate_unique_config(config):
    config["model"] = config["model"]["type"]

    keys, values = zip(*config.items())  # Extract keys and values
    combinations = [dict(zip(keys, v)) for v in product(*values)]  # Create dicts for each combination
    return combinations

def main():
    configs = loading_data.load_config_file('configs/r4_config_learning_curve_test.yaml')
    configs = generate_unique_config(configs)

    for config in configs:
    
        lc = LearningCurve(config,random_seed_id=2,optimization='selected_best_random_100')

        train_sizes, train_scores, test_scores = lc.compute_learning_curve()


    
    

if __name__ == '__main__':

    main()