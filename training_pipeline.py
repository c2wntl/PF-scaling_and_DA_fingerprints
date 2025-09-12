from src import training_predictor
from src import loading_data,path_manipulation,model_evaluation
from src.utils import plot,DataHandler,ResultsHandler
import json,os,time,joblib,re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from feature_engineering import DimensionalReduction

class training_pipeline:
    def __init__(self,model_name,X_train_transform,X_test_transform,
                 y_train,y_test,features,save_path,weights=None):
        ## declaring the global pipeline input
        self.model_name = model_name
        self.X_train_transform = X_train_transform
        self.X_test_transform = X_test_transform
        self.y_train = y_train
        self.y_test = y_test
        self.features = features
        self.save_path = save_path
        self.weights = weights

        ## the global variable
        kfold_split = 5
        kfold_random_state = 42
        self.kfold = KFold(n_splits=kfold_split,shuffle=True,
                      random_state=kfold_random_state)

    def blind_test_save_results(self,model,X_train_transform,X_test_transform,
                                y_train,y_test,save_path,features,weights=None):

        training_time_start = time.time()
        model.fit(X_train_transform, y_train,sample_weight = weights.loc[y_train.index])
        training_time_end = time.time()

        # calculate the training time
        training_time = training_time_end - training_time_start

        # save the model
        joblib.dump(model,os.path.join(save_path,'model.pkl'))

        # save the blind test results
        ResultsHandler.save_blind_test_results(model,features,X_train_transform,
                                               X_test_transform,y_train,y_test,
                                               save_path,weights
                                               )

        return training_time

    def base_model(self,feature_engi_time,initiate_model=None):
        ## initialize the model
        if initiate_model is None:
            model = training_predictor.initialize_model(self.model_name)
        else:
            model = initiate_model
        
        ## perform cross validation
        cv_score, cv_time = training_predictor.performing_cv(
            kfold=self.kfold, model=model, 
            X_train=self.X_train_transform, 
            y_train=self.y_train, weights=self.weights
        )

        ## save the cv results
        ResultsHandler.cv_results(cv_score,self.save_path)

        ## perform the blind test set and save the results
        training_time = self.blind_test_save_results(model,self.X_train_transform,self.X_test_transform,
                                     self.y_train,self.y_test,self.save_path,
                                     self.features,weights=self.weights
                                     )
        
        ## save the splitted data
        ResultsHandler.save_splitting_data(self.save_path,self.weights,self.features,
                                           self.X_train_transform,self.X_test_transform,
                                           self.y_train,self.y_test
                                           )

        ## save the operated time 
        pd.DataFrame({'optimized_time':[None],
                      'training_time':[training_time],
                      'cv_time':[cv_time]}).to_csv(os.path.join(self.save_path,'time.csv'))
        ResultsHandler.save_feature_engineering_time(feature_engi_time,self.save_path)



            

    def selected_optimized_model(self,feature_engi_time,results_params_dir,pattern =  'results_(\d+)',params_dir='best_random',original_best_params=True,mae_weight = 0.6):
        match = re.search(pattern,self.save_path)
        if match:
            params_path = re.sub(pattern,results_params_dir,self.save_path) 
            if original_best_params:
                params_path = os.path.join(params_path,params_dir,'best_params.json')
                ### load the optimized parameters
                params = json.load(open(params_path,'r'))
                save_path = os.path.join(self.save_path,f'selected_{params_dir}')
            else:
                search_results_path = os.path.join(params_path,params_dir,'search_results.csv')
  
                search_results = pd.read_csv(search_results_path,index_col=0)
                base_model_results = pd.read_csv(os.path.join(params_path,'eval_results_from_cv.csv'),index_col=0)
                base_model_results = ResultsHandler.calculate_test_train_ratio(base_model_results,
                                                        test_col='test_weighted_mae_mean',train_col='train_weighted_mae_mean',
                                                        error_test_col='test_weighted_mae_std',error_train_col='train_weighted_mae_std',) 
                params, _, _ = ResultsHandler.select_params(search_results,mae_weight=mae_weight,
                                                                                        base_model_mae= base_model_results['test_weighted_mae_mean'].iloc[0])
                save_path = os.path.join(self.save_path,f'selected_{params_dir}_{int(mae_weight * 100)}')



        ## redirect the savepath
        os.makedirs(save_path,exist_ok=True)

        
        ## initialize the model
        model = training_predictor.initialize_model(self.model_name,params=params)
        
        ## perform cross validation
        cv_score, cv_time = training_predictor.performing_cv(
            kfold=self.kfold, model=model, 
            X_train=self.X_train_transform, 
            y_train=self.y_train, weights=self.weights
        )

        ## save the cv results
        ResultsHandler.cv_results(cv_score,save_path)

        ## perform the blind test set and save the results
        training_time = self.blind_test_save_results(model,self.X_train_transform,self.X_test_transform,
                                     self.y_train,self.y_test,save_path,
                                     self.features,weights=self.weights
                                     )
        
        pd.DataFrame({'optimized_time':[None],
                      'training_time':[training_time],
                      'cv_time':[cv_time]}).to_csv(os.path.join(save_path,'time.csv'))
        ResultsHandler.save_feature_engineering_time(feature_engi_time,save_path)

        ## save the optimized parameters and its results_params_dir
        meta_data = {'optimized_params':params,
                     'results_params_dir':results_params_dir,
                     'params_dir':params_dir}
        with open(os.path.join(save_path,'meta_data.json'),'w') as json_file:
            json.dump(meta_data,json_file,indent=4)

    def optimize(self,search_type,param_grid,n_iter=1):

        ## changing the save path
        if search_type == 'random_search':
            save_path = os.path.join(self.save_path,'best_random')
        elif search_type == 'grid_search':
            save_path = os.path.join(self.save_path,'gross_grid')
        os.makedirs(save_path,exist_ok=True)

        ## initialize the model
        initial_model = training_predictor.initialize_model(self.model_name,n_jobs=4)

        ## start random grid search
        optimized_time_start = time.time()
        if search_type == 'random_search':
            _, random_results_df, _ = training_predictor.random_search(self.kfold,initial_model,param_grid, 
                                                                                                            self.X_train_transform,self.y_train,
                                                                                                            n_jobs=1,
                                                                                                            weights=self.weights,
                                                                                                            n_iter=n_iter)
        elif search_type == 'grid_search':
            _, random_results_df, _ = training_predictor.grid_search(self.kfold,initial_model,param_grid,
                                                                                                            self.X_train_transform,self.y_train,
                                                                                                            n_jobs=4,
                                                                                                            weights=self.weights,

            )
        else:
            raise ValueError(f"Invalid search_type: {search_type}. Must be 'random_search' or 'grid_search'.")

        ## This line is to get the best parameters and results from the search with my criteria, i.e. the mean_weighted_mae and the test_train_ratio
        best_random_param, random_results_df, best_random_results_df = ResultsHandler.get_best_params_and_search_results(random_results_df)
        
        optimized_time_end = time.time()                                                                                                    

        
        ## manipulate and save the search results
        best_random_results_df = ResultsHandler.convert_search_best_score(best_random_results_df)
        
        random_results_df.to_csv(os.path.join(save_path,'search_results.csv'))
        best_random_results_df.to_csv(os.path.join(save_path,'eval_results_from_cv.csv'))

        with open(os.path.join(save_path,'best_params.json'),'w') as json_file:
            json.dump(best_random_param, json_file, indent=4)
        
        ## use the optimized parameter to the model for blind test
        model = training_predictor.initialize_model(self.model_name,params=best_random_param,n_jobs=6)
        training_time = self.blind_test_save_results(model, self.X_train_transform,
                                                    self.X_test_transform,
                                                    self.y_train,self.y_test,
                                                    save_path,self.features,
                                                    weights=self.weights
                                                    )

        plot.bar_compare_train_test(best_random_results_df,os.path.join(save_path,'train_test_metrics.png'),best_score=True)

        ## save the calculated time
        optimized_time = optimized_time_end - optimized_time_start
        pd.DataFrame({'optimized_time':[optimized_time],
                      'training_time':[training_time],
                      'cv_time':[None]}).to_csv(os.path.join(save_path,'time.csv'))

        

class data_utils:
    """
    Utility class for loading datasets and generating weights for training.

    This class provides methods to load high-density, low-density, or combined
    datasets and compute corresponding sample weights and base densities using
    the DataHandler.
    """
    
    def __init__(self,database,dataset,):
        self.database = database
        self.dataset = dataset
        

    def full_dataset(self):
        if  self.dataset == 'all_density':
            full_high_df =loading_data.full_data('2dmd','all_high_density','cfid')
            full_low_df =loading_data.full_data('2dmd','all_low_density','cfid')

            full_df,weights,base_density = DataHandler.generate_weights(full_high_df,full_low_df)
            
        elif self.dataset == 'all_high_density':
            full_high_df =  loading_data.full_data(self.database,self.dataset,'cfid')
            full_df,weights,base_density = DataHandler.generate_weights(full_high_df)

        elif self.dataset == 'all_low_density':
            full_low_df =  loading_data.full_data(self.database,self.dataset,'cfid')
            full_df, weights,base_density = DataHandler.generate_weights(full_low_df)

        else:
            raise ValueError(f"{self.dataset} is not defined")
            
        return  full_df, weights, base_density



def main_pipeline(database:str,model_name:str,target:str,
              dataset:str,feature_set:str,random_seed_id=None,
              pca_feature_sets = ['pca_cfid']):
    ################################# preparing the data #################################             
    ### Declare the parameter grids 
    param_grids = {
                    "CatBoostRegressor":{
                                         'learning_rate': [ 0.1, 0.03, 0.01, 0.005, 0.001],
                                         'l2_leaf_reg': [ 1, 3, 5, 7, 9],
                                         'border_count': [ 128, 254],
                                         'grow_policy': [ 'SymmetricTree', 'Depthwise', 'Lossguide'],
                                         'min_data_in_leaf': [ 1, 5, 10],
                                         'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
                                        }  
                }

    param_grid = param_grids[model_name]
    
    save_path = path_manipulation.get_save_path(database,model_name,
                                                target,feature_set,
                                                dataset,random_seed_id=random_seed_id)
    
    ### Load the data
    print('====================================================== preparing the dataset ======================================================')
    data_pre = data_utils(database = database,dataset = dataset)
    print('-------------> 01 load the data')
    full_df, weights, base_density = data_pre.full_dataset() 
    bases = full_df['base']

    X,y = loading_data.split_X_y(full_data_df=full_df,feature_set = 'cfid', target = target)

    ### Engineering the features and data splitting
    print('-------------> 02 engineer the features')
    feature_engineering_time_start = time.time()
    dr = DimensionalReduction(X,y, how = feature_set, 
                              bases = bases, base_density=base_density,
                              random_seed_id=random_seed_id,
                              standard_scaler=False)

    X_train_transform = dr.X_train_transform
    X_test_transform = dr.X_test_transform
    y_train = dr.y_train_transform
    y_test = dr.y_test_transform
    features = dr.features

    feature_engineering_time_end = time.time()
    
    ################################# training start here #################################             
    print('====================================================== start training session ======================================================')
    pipe = training_pipeline(model_name,X_train_transform,X_test_transform,
                             y_train,y_test,features,save_path,weights=weights)


    print('-------------> 03 train the default CatBoost model')
    #------- train and evaluate the default CatBoost model
    pipe.base_model(feature_engi_time=feature_engineering_time_end-feature_engineering_time_start)


    print('-------------> 04 perform the randomized search')
    #------- tune the model with hyperparameters
    pipe.optimize(search_type='random_search',param_grid=param_grid,n_iter=3) # randomized search CV
    # pipe.optimize(search_type='grid_search',param_grid=param_grid) # the option of using grid search is also available

    
    
    ################################ select the tuned model #################################
    if random_seed_id is not None:
        results_params_dir = f'results_{random_seed_id}' # selecting the hyperparameter should be done separately to the optimization process
    else:
        results_params_dir = 'results' 

    # the optional choices for selecting the model based on the weight given to MAE 
    # for mae_weight in [0.6,0.7,0.8,0.9,0.99,1.0]:
    #     pipe.selected_optimized_model(feature_engi_time=feature_engineering_time_end-feature_engineering_time_start,
    #                                 results_params_dir=results_params_dir,params_dir='best_random',
    #                                 original_best_params=False,mae_weight=mae_weight)
    

    print('-------------> 05 select the model')
    pipe.selected_optimized_model(feature_engi_time=feature_engineering_time_end-feature_engineering_time_start,
                                  results_params_dir=results_params_dir,params_dir='best_random',original_best_params=False,mae_weight=1.0)

    
    
if __name__ == '__main__':
    import argparse

    
    # Create the parser
    parser = argparse.ArgumentParser(description="Process command-line arguments")
    
    # Add arguments
    parser.add_argument('--database', required = True, help="Database name, e.g., '2dmd'")
    parser.add_argument('--model_name', required=True, help="Name of the model")
    parser.add_argument('--target', required=True, help="Target name")
    parser.add_argument('--feature_set', required=True, help="Feature set to use")
    parser.add_argument('--dataset', required=True, help="Dataset name")
    parser.add_argument('--random_seed_id', required=True, help="random seed id")
    
    args = parser.parse_args()
    
    database = args.database
    model_name = args.model_name
    target = args.target
    feature_set = args.feature_set
    dataset = args.dataset
    random_seed_id = args.random_seed_id
    
    main_pipeline(database,model_name,target,dataset,feature_set,random_seed_id=random_seed_id)
