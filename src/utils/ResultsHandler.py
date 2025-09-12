from src import model_evaluation
from src.utils import plot
from src import loading_data
import numpy as np 
import pandas as pd
import os,re,ast

class LoadFromResults:

    def _get_dir_path():
        project_path = os.path.dirname(os.path.realpath(f'{__file__}/../..'))
        print(project_path)
    
    

def cv_results(cv_score:dict,save_path):
    """
    Save the eval_results_from_cv.csv and the train_test bar plot (train_test_metrics.png)
    """
    model_evaluation.get_eval_results_from_cv(cv_score,os.path.join(save_path,'eval_results_from_cv.csv'))
    # plot.bar_compare_train_test(cv_score,os.path.join(save_path,'train_test_metrics.png'))

def save_splitting_data(save_path,weights,features,X_train_transform,X_test_transform,y_train,y_test):
    if isinstance(X_train_transform, np.ndarray):
        X_train_transform = pd.DataFrame(data=X_train_transform,columns=features,index=y_train.index)
        X_test_transform = pd.DataFrame(data=X_test_transform,columns=features,index=y_test.index)

    X_train_transform.to_csv(os.path.join(save_path, 'X_train.csv'))#, index=False)
    X_test_transform.to_csv(os.path.join(save_path, 'X_test.csv'))#, index=False)
    y_train.to_csv(os.path.join(save_path, 'y_train.csv'))#, index=False)
    y_test.to_csv(os.path.join(save_path, 'y_test.csv'))#, index=False)
    weights.to_csv(os.path.join(save_path,'sample_weight.csv'))

def save_blind_test_results(model,features,X_train_transform,
                            X_test_transform,y_train,y_test,
                            save_path,weights):
    blind_eval_results,y_pred_test = model_evaluation.get_eval_results(model,X_train_transform,X_test_transform,y_train,y_test,SAVEPATH=os.path.join(save_path,'eval_metrics.csv'),weights=weights)
    # MAE_test = blind_eval_results['MAE_test'].iloc[0]
    # R2_test = blind_eval_results['R2_test'].iloc[0]
    _ = model_evaluation.get_feature_importance(model,features,SAVEPATH=os.path.join(save_path,"feature_importance.csv"))
    # plot.feature_importance(model,features,SAVEPATH = os.path.join(save_path,'feature_importance.png'))
    # plot.parity(y_pred_test,y_test,TITLE= f'MAE: {MAE_test:.4f}, r2: {R2_test:.4f}',SAVEPATH = os.path.join(save_path,'parity_test.png'))

def convert_search_best_score(best_results):
    """    Converts the keys in the best_results dictionary to a more readable format 
    and negates the MAE mean values to reflect the true MAE.

    Args:
        best_results (_type_): _description_

    Returns:
        _type_: _description_
    """
    results_T_df = pd.DataFrame(best_results).T
    original_pattern = r'(mean|std)_(train|test)_(r2|neg_mean_absolute_error|weighted_mae)'
    rename_metrics = {'r2':'r2',
                        'neg_mean_absolute_error':'mae',
                        'weighted_mae':'weighted_mae'}
    new_column_names = []
    original_column_names =[]
    for column_name in results_T_df.columns:
        match = re.search(original_pattern,column_name)
        if match:
            aggre = match.group(1)
            data_set = match.group(2)
            metric = match.group(3)
            original_column_names.append(column_name)
            new_column_names.append(f"{data_set}_{rename_metrics[metric]}_{aggre}")

    rename_dict = {key:value for key, value in zip(original_column_names,new_column_names) }

    results_T_df =  results_T_df[rename_dict.keys()].rename(columns=rename_dict)
    for column in results_T_df.columns:
        if ('mae_mean' in column) and (results_T_df.loc[:,column].mean() < 0):
            results_T_df.loc[:,column] = -results_T_df.loc[:,column].values

    
    return results_T_df

def save_feature_engineering_time(feature_engi_time,save_path):
    pd.DataFrame({'feature_engineering_time': [feature_engi_time]}).to_csv(os.path.join(save_path,'feature_engineering_time.csv'))

    
def select_methods(sep_df,chem_part=None,en_chem=None,dist_part=None,distance=None):

    is_chem_part = False
    is_en_chem = False
    is_alldist = False
    is_distance = False
    
    if chem_part is not None:
        is_chem_part = True

    if en_chem is not None:
        is_en_chem=True
    if dist_part is not None:
        is_alldist = True
    if distance is not None:
        is_distance = True

        
    if is_chem_part:
        sep_df = sep_df[sep_df['chem_part']==chem_part]
        
    if is_en_chem:
       if en_chem == 'vpa':
            print(en_chem)
            # sep_df[ sep_df['en_chem'].values.isin([f for f in sep_df['en_chem'].values if f.startswith('vpa')])]
            sep_df = sep_df.loc[ sep_df['en_chem'].isin([f for f in sep_df['en_chem'].values if f.startswith(en_chem)]),:]
       else:
            sep_df = sep_df[sep_df['en_chem']==en_chem]
    
    if is_alldist:
       sep_df = sep_df[sep_df['dist_part']==dist_part]

    if is_distance:

       sep_df = sep_df[sep_df['distance']==distance]
    return sep_df.copy()


def split_feature_set_name(f):
    distance_mapping = {'hellinger':'Hellinger','tvd':'TVD','emd':'EMD'}
    en_chem_mapping = {'mult':'Multiplication','divi':'Division','subs':'Subtraction'}
    en_chem_name = None
    distance_name = None
    if 'vpa_' in f:
        en_chem_name = f'PF-{en_chem_mapping[f.split("_")[1]]}'

    if 'pristine' in f:
        pattern = r"(hellinger|jsd|emd|tvd)_"
        match = re.search(pattern,f)
        if match:
            distance_name = distance_mapping[match.group(1)]

    if 'alldist' in f:
        include_full_dist = True
    elif 'pristine' in f and 'all_dist' not in f:
        include_full_dist = False
    elif 'dist0' in f:
        include_full_dist = False
    else:
        include_full_dist = True

    feature_label = ''
    if include_full_dist:
        feature_label += f'Including the full Distr. \n '
    else:
        feature_label += f'Excluding the full Distr. \n '

    if en_chem_name is not None:
        feature_label += en_chem_name + " "
    if en_chem_name is not None and distance_name is not None:
        feature_label += "with "
    if distance_name is not None:
        feature_label += distance_name
    if en_chem_name is None and distance_name is None:
        
        feature_label += "Original CFID"
    return feature_label 

    
def transform_feature_set_column(df,include_feature_name=False):
    results = df[['feature_set']]
    chem_context = []
    engineer_chem = []
    engineer_dist = []
    all_dist = []
    for f in results['feature_set']:
        if 'eo_chem' in f:
            chem_context.append('eo_chem')
        else:
            chem_context.append('chem')

        if 'vpa_' in f:
            engineer_chem.append(f'vpa_{f.split("_")[1]}')
        else:
            engineer_chem.append('no')

        if 'pristine' in f:
            pattern = r"(hellinger|jsd|emd|tvd)_"
            match = re.search(pattern,f)
            if match:
                engineer_dist.append(match.group(1))
            else:
                print(f)
                engineer_dist.append('error')
        else:
            engineer_dist.append('no')

        if 'alldist' in f:
            all_dist.append('alldist')
        elif 'pristine' in f and 'all_dist' not in f:
            all_dist.append('no')
        elif 'dist0' in f:
            all_dist.append('no')
        else:
            all_dist.append('alldist')
            
            
    separate_feature_set = pd.DataFrame({'chem_part':chem_context,
                                        'dist_part': all_dist,
                                        'en_chem': engineer_chem,
                                        'distance': engineer_dist,})
                                        # 'mean_MAE_test': target_results['mean_MAE_test'],
                                        # 'mean_MAE_train':target_results["mean_MAE_train"],
                                        # 'std_MAE_test': target_results['std_MAE_test'],
                                        # 'std_MAE_train':target_results["std_MAE_train"],
                                        # 'test_train_ratio': target_results['mean_MAE_test']/target_results['mean_MAE_train']
                                        # })
    # separate_feature_set.set_index(df.index,inplace=True)
    if include_feature_name:
        return pd.merge(separate_feature_set,df,left_index=True,right_index=True)
    else:
        return pd.merge(separate_feature_set,df.drop(columns=['feature_set']),left_index=True,right_index=True)

        
def get_euclidean(data,column1,column2,mae_weight = 0.6):
    """
    Calculate the Euclidean distance for each row in the DataFrame based on two specified columns.

    This function normalizes the values in the specified columns, computes a reference point for the second column,
    and then calculates a weighted Euclidean distance for each row. The result is stored in a new column named 'euclidean'.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.
    column1 (str): The name of the first column to be used in the distance calculation.
    column2 (str): The name of the second column to be used in the distance calculation.

    Returns:
    pd.DataFrame: The DataFrame with an additional column 'euclidean' containing the calculated distances.
    """
    dummy_list = []
    min_c1 = data[column1].min()
    max_c1 = data[column1].max()

    min_c2 = data[column2].min()
    max_c2 = data[column2].max()

    ref_point = (1-min_c2)/(max_c2 - min_c2)
    ratio_weight = 1-mae_weight
    for i in data.index:
        x = (data.loc[i,column1]-min_c1) / (max_c1 - min_c1)
        y = (data.loc[i,column2]-min_c2) /(max_c2 - min_c2) 
        dummy_list.append(np.sqrt( mae_weight* (x)**2  + ratio_weight* (y-ref_point)**2))

    data['euclidean'] = dummy_list
    return data


def get_best_params_and_search_results(search_results:pd.DataFrame):
    """
    Processes the search results DataFrame to find the best parameters and search results.

    This function performs the following steps:
    1. Converts the 'mean_test_weighted_mae' and 'mean_train_weighted_mae' columns to positive values.
    2. Sorts the DataFrame by 'rank_test_weighted_mae'.
    3. Calculates the ratio of 'mean_test_weighted_mae' to 'mean_train_weighted_mae' and stores it in a new column 'mean_test_train_mae_ratio'.
    4. Computes the Euclidean distance between 'mean_test_weighted_mae' and 'mean_test_train_mae_ratio' and stores it in a new column 'euclidean'.
    5. Sorts the DataFrame by 'euclidean' in ascending order.
    6. Assigns a new ranking based on the sorted 'euclidean' values and stores it in 'my_rank_test_weighted_mae'.
    7. Extracts the best results (first row of the sorted DataFrame).
    8. Parses the 'params' column of the best results to get the best parameters.

    Args:
        search_results (pd.DataFrame): The DataFrame containing the search results.

    Returns:
        tuple: A tuple containing:
            - best_params (dict): The best parameters extracted from the search results.
            - search_results (pd.DataFrame): The modified search results DataFrame.
            - best_results (pd.Series): The best results row from the search results DataFrame.
    """
    search_results['mean_test_weighted_mae'] = search_results['mean_test_weighted_mae'] * -1
    search_results['mean_train_weighted_mae'] = search_results['mean_train_weighted_mae'] * -1
    search_results.sort_values(by='rank_test_weighted_mae')[['rank_test_weighted_mae','mean_test_weighted_mae','mean_train_weighted_mae']]
    search_results['mean_test_train_mae_ratio'] = search_results['mean_test_weighted_mae'] / search_results['mean_train_weighted_mae']
    search_results = get_euclidean(data= search_results, column1= 'mean_test_weighted_mae',column2= 'mean_test_train_mae_ratio')
    search_results.sort_values(by='euclidean',ascending=True,inplace=True)
    search_results['my_rank_test_weighted_mae'] = range(1,len(search_results)+1)

    best_results = search_results.iloc[0]

    best_params = best_results['params'] 
    # print(type(best_results['params']))

    # best_params = ast.literal_eval(best_results['params'])

    return best_params, search_results, best_results

def select_params(search_results:pd.DataFrame,base_model_mae,mae_weight=0.6, threshold_ratio = 1.5):

    search_results['mean_test_weighted_mae'] = search_results['mean_test_weighted_mae'] 
    search_results['mean_train_weighted_mae'] = search_results['mean_train_weighted_mae'] 
    search_results.sort_values(by='rank_test_weighted_mae')[['rank_test_weighted_mae','mean_test_weighted_mae','mean_train_weighted_mae']]
    search_results['mean_test_train_mae_ratio'] = search_results['mean_test_weighted_mae'] / search_results['mean_train_weighted_mae']
    search_results['std_test_train_mae_ratio'] = search_results['mean_test_train_mae_ratio'] * np.sqrt(
        (search_results['std_test_weighted_mae']/search_results['mean_test_weighted_mae'])**2 + (search_results['std_train_weighted_mae']/search_results['mean_train_weighted_mae'])**2
    )
    search_results = get_euclidean(data= search_results, column1= 'mean_test_weighted_mae',column2= 'mean_test_train_mae_ratio',mae_weight=mae_weight)
    search_results.sort_values(by='euclidean',ascending=True,inplace=True)
    search_results['my_rank_test_weighted_mae'] = range(1,len(search_results)+1)
    search_results['res'] = search_results['mean_test_weighted_mae'] - base_model_mae 
    if threshold_ratio is None:
        search_results = search_results[(search_results['res']<0)]
    else:
        search_results = search_results[(search_results['res']<0) & (search_results['mean_test_train_mae_ratio']<threshold_ratio)]

    # print(search_results)
    best_results = search_results.iloc[0]
    # print( type(best_results['params']))

    best_params = ast.literal_eval(best_results['params'])
    # print(type(best_params))

    return best_params, search_results, best_results
    

def calculate_aggregate_results(df:pd.DataFrame,test_metric='MAE_test',train_metric='MAE_train'):
    filt_df = df[['target_column','feature_set','random_seed_id',test_metric,train_metric]]
    features = np.unique(filt_df['feature_set'])
    targets = np.unique(filt_df['target_column'])
    results = []
    for target in targets:
        for feature in features:
            temp_filt_df = filt_df[(filt_df.feature_set == feature) & (filt_df.target_column == target)]
            results.append({
                        'target_column':target,
                        'feature_set':feature,
                        'mean_MAE_test':temp_filt_df[test_metric].mean(),
                        'std_MAE_test':temp_filt_df[test_metric].std(),
                        'mean_MAE_train':temp_filt_df[train_metric].mean(),
                        'std_MAE_train':temp_filt_df[train_metric].std(),
                       
                        })

    results_df = pd.DataFrame(results)
    
    return results_df

def calculate_test_train_ratio(data,test_col='mean_MAE_test',train_col='mean_MAE_train',error_test_col='std_MAE_test',error_train_col='std_MAE_train'):
    # display(data)
    data['test_train_ratio'] = data[test_col] / data[train_col]
    data['std_test_train_ratio'] = data['test_train_ratio'] * np.sqrt(
            (data[error_test_col] / data[test_col])**2 +
            (data[error_train_col] / data[train_col])**2
        )
    return data

