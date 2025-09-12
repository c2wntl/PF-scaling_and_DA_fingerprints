
import os,sys,yaml,re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import itertools
import ast
import seaborn as sn
from pymatgen.core.periodic_table import Element

from jarvis.analysis.structure.neighbors import calc_structure_data

this_file_path = os.path.abspath(__file__)
notebooks_path = os.path.join(this_file_path,'..')

project_path = os.path.abspath('..')
src_path = os.path.abspath(os.path.join('..', 'src'))
utils_path = os.path.join(src_path,'utils')

if utils_path not in sys.path:
    sys.path.append(utils_path)

if src_path not in sys.path:
    sys.path.append(src_path) 

import loading_data,analyzing_data

with open(os.path.join(project_path,'configs','config_colors.yaml'),'r') as file:
    config_colors = yaml.safe_load(file)
with open(os.path.join(project_path,'configs','config_labels.yaml'),'r') as file:
    config_labels = yaml.safe_load(file)
with open(os.path.join(project_path,'configs','config_pattern.yaml'),'r') as file:
    config_patterns = yaml.safe_load(file)
    
def getcolor(feature_set):
    if 'pca_cfid' in feature_set:
        color = config_colors['feature_set']['pca_cfid']
    else:
        # print(config_colors)
        color= config_colors['feature_set'][feature_set]
    return color
    
def gethatch(feature_set):
    if 'pca_cfid' in feature_set:
        hatch = config_patterns['feature_set']['pca_cfid']
    else:
        hatch = config_patterns['feature_set'][feature_set]
    return hatch

def get_cv_results(config,optimize=None,result_dirname='results'):
    config_info_df = pd.DataFrame({'database':[],'model':[],'target_column':[],'feature_set':[],'dataset':[]})
    cv_result_df = pd.DataFrame({'train_r2_mean': [],
                                'test_r2_mean': [],
                                'train_r2_std': [],
                                'test_r2_std': [],
                                'train_mae_mean': [],
                                'test_mae_mean': [],
                                'train_mae_std': [],
                                'test_mae_std': []})
    blind_test_result_df = pd.DataFrame({'R2_train': [],
                                        'R2_test': [],
                                        'MAE_train': [],
                                        'MAE_test': [],
                                        'MSE_train': [],
                                        'MSE_test': []
                                        })
    for database in config['database']:
        for model in config['model']['type']:  # Access model types correctly
            for target_column in config['target_column']:
                for feature_set in config['feature_set']:
                    for dataset in config['dataset']:
                        filename = 'eval_results_from_cv.csv'
                        try:
                            cv_result_row = loading_data.load_results(filename,database,feature_set,dataset,target_column,model,optimize,result_dirname=result_dirname)

                            blind_test_result_row = loading_data.load_results('eval_metrics.csv',database,feature_set,dataset,target_column,model,optimize,result_dirname=result_dirname)
                        
                            time_row = loading_data.load_results('time.csv',database,feature_set,dataset,target_column,model,optimize,result_dirname=result_dirname)
                        except FileNotFoundError as e:
                            print(f"{e}")


                        config_info_row = pd.DataFrame({'database':[database],'model':[model],'target_column':[target_column],'feature_set':[feature_set],'dataset':[dataset]})

                        config_info_row = pd.merge(config_info_row,time_row,left_index=True,right_index=True)

                        cv_result_df = pd.concat([cv_result_df,cv_result_row],ignore_index=True)
                        blind_test_result_df = pd.concat([blind_test_result_df,blind_test_result_row],ignore_index=True)
                        config_info_df = pd.concat([config_info_df,config_info_row],ignore_index=True)


                        
    final_df =  pd.merge(config_info_df,cv_result_df,how='outer',left_index=True,right_index=True)
    final_df =  pd.merge(final_df,blind_test_result_df,how='outer',left_index=True,right_index=True)
    # display(final_df)
    return final_df

def get_feature_importances(config,optimize=None,N=15,result_dirname='results'):
    config_info_df = pd.DataFrame({'database':[],'model':[],'target_column':[],'feature_set':[],'dataset':[]})
    main_df = pd.DataFrame()
    for database in config['database']:
        for model in config['model']['type']:
            for feature_set in config['feature_set']:
                for dataset in config['dataset']:
                    for target_column in config['target_column']:
                        filename = 'feature_importance.csv'

                        feature_importances = loading_data.load_results(filename,database,feature_set,dataset,target_column,model,optimize,result_dirname)
                        if N is not None:
                            feature_importances = feature_importances.head(N)
                        fi_1 = feature_importances.set_index('Feature').T.reset_index(drop=True) 
                        # fi_1 = fi_1.divide(fi_1.sum(axis=1).iloc[0])

                        main_df = pd.concat([main_df,fi_1],ignore_index=True)

                        config_info_row = pd.DataFrame({'database':[database],'model':[model],'target_column':[target_column],'feature_set':[feature_set],'dataset':[dataset]})
                        config_info_df = pd.concat([config_info_df,config_info_row],ignore_index=True)

    main_df = pd.merge(config_info_df,main_df,left_index=True,right_index=True,how='outer').fillna(0)

    return main_df


def calculate_test_train_ratio_std(test,std_test,train,std_train):
    F = test/train
    return F*(std_test/test * std_train/train)
def plot_train_test_compare(results_df,score1:str='test_mae_mean',score2:str='train_mae_mean',operator='ratio',title="Train-Validate Ratio",):
    target_columns = np.unique(results_df['target_column'])
    datasets = np.unique(results_df['dataset'])
    nrows = len(target_columns)
    ncols = len(datasets)
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10), sharey=False)
    
    total_group_width = 0.1
    feature_sets = results_df['feature_set'].drop_duplicates()
    models = results_df['model'].drop_duplicates()
    n_bars = len(feature_sets)
    
    bar_width = total_group_width / n_bars
    indices = np.arange(len(models))

    for irow, target_column in enumerate(target_columns):
        for icol, dataset in enumerate(datasets):
            df = results_df[(results_df['target_column'] == target_column) & (results_df['dataset'] == dataset)]
            ax = axs[irow, icol] if nrows > 1 else axs[icol]
            
            # Ensure there is data to plot
            if df.empty:
                continue
            
            for i, model in zip(np.arange(0,(len(models))),models):
                # i=i+1
                i=i*2
                subset_df = df[df['model'] == model]
                if subset_df.empty:
                    continue
                j=0
                for _, feature_set in enumerate(feature_sets):
                    feature_subset = subset_df[subset_df['feature_set'] == feature_set]
                    if feature_subset.empty:
                        continue
                    j = j+1
                    
                    # Calculate bar positions: now adjusted to group by model and feature set
                    bar_positions = i * (n_bars * bar_width) + j * bar_width 

                    test_mae_mean = feature_subset['test_mae_mean'].values
                    test_mae_std = feature_subset['test_mae_std'].values
                    train_mae_mean = feature_subset['train_mae_mean'].values
                    train_mae_std = feature_subset['train_mae_std'].values
                    
                    color = getcolor(feature_set)
                    hatch = gethatch(feature_set)

                    if operator == 'ratio':
                        height =  feature_subset[score1].values/feature_subset[score2].values
                    elif operator == 'subtract':
                        height =  feature_subset[score1].values-feature_subset[score2].values
                    else:
                        ValueError(f'the operator {operator} is not defined') 
                    # Plot bars for test MAE with error bars
                    ax.bar(bar_positions, 
                            height,
                           width=bar_width, 
                           yerr=calculate_test_train_ratio_std(test_mae_mean,test_mae_std,train_mae_mean,train_mae_std), 
                           capsize=5, 
                           edgecolor='black',
                           color=color,
                           hatch=hatch,
                           label=feature_set if i == 0 else "")
            ax.grid()
            ax.set_title(f'{target_column} - {dataset}')
            ax.set_xticks(indices*2 * n_bars * bar_width)
            ax.set_xticklabels(models, rotation=0,ha='left')
            ax.set_ylabel('CV-score (MAE)')
            ax.legend(title='Feature Set')
    plt.suptitle(title,fontsize=18)
    plt.tight_layout()
    plt.show()
    
def plot_blind_test(results_df,title="MAE of the Model on Blind Test Set"):
    target_columns = np.unique(results_df['target_column'])
    datasets = np.unique(results_df['dataset'])
    nrows = len(target_columns)
    ncols = len(datasets)
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10), sharey=False)
    
    total_group_width = 0.1
    feature_sets = results_df['feature_set'].drop_duplicates()
    models = results_df['model'].drop_duplicates()
    n_bars = len(feature_sets)
    
    bar_width = total_group_width / n_bars
    indices = np.arange(len(models))

    for irow, target_column in enumerate(target_columns):
        for icol, dataset in enumerate(datasets):
            df = results_df[(results_df['target_column'] == target_column) & (results_df['dataset'] == dataset)]
            ax = axs[irow, icol] if nrows > 1 else axs[icol]
            
            # Ensure there is data to plot
            if df.empty:
                continue
            
            for i, model in zip(np.arange(0,(len(models))),models):
                # i=i+1
                i=i*2
                subset_df = df[df['model'] == model]
                j=0
                if subset_df.empty:
                    continue
                
                for _, feature_set in enumerate(feature_sets):
                    feature_subset = subset_df[subset_df['feature_set'] == feature_set]
                    if feature_subset.empty:
                        continue
                    j=j+1
                    # Calculate bar positions: now adjusted to group by model and feature set
                    bar_positions = i * (n_bars * bar_width) + j * bar_width 

                    if 'pca' in feature_set:
                        color='#2ca02c'
                        hatch='.'
                    else:
                        color=config_colors['feature_set'][feature_set]
                        hatch=config_patterns['feature_set'][feature_set]
                    # print(feature_subset['MAE_test'])
                    # Plot bars for test MAE with error bars
                    ax.bar(x=bar_positions, 
                           height=feature_subset['MAE_test'].values, 
                           width=bar_width, 
                           #yerr=calculate_test_train_ratio_std(test_mae_mean,test_mae_std,train_mae_mean,train_mae_std), 
                          # capsize=5, 
                           edgecolor='black',
                           color=color,
                           hatch=hatch,
                           label=feature_set if i == 0 else "")
                    ax.bar(bar_positions, 
                           feature_subset['MAE_train'].values, 
                           width=bar_width, 
                           capsize=5, 
                           edgecolor='black',
                           color='powderblue',
                           hatch=hatch,)
            ax.grid()
            ax.set_title(f'{target_column} - {dataset}')
            ax.set_xticks(indices*2 * n_bars * bar_width)
            ax.set_xticklabels(models, rotation=0,ha='left')
            ax.set_ylabel('Test MAE')
    plt.legend(title='Feature Set',ncols=2,bbox_to_anchor=(0.5,-0.1))

    plt.suptitle(title,fontsize=18)
    # plt.tight_layout()
    plt.show()
  
def plot_cv_mae_mean(results_df,title="Average MAE from Cross-Validation"):
    target_columns = np.unique(results_df['target_column'])
    datasets = np.unique(results_df['dataset'])
    nrows = len(target_columns)
    ncols = len(datasets)
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10), sharey=False)
    total_group_width = 0.1
    feature_sets = results_df['feature_set'].drop_duplicates()
    models = results_df['model'].drop_duplicates()
    n_bars = len(feature_sets)
    
    bar_width = total_group_width / n_bars
    indices = np.arange(len(models))

    for irow, target_column in enumerate(target_columns):
        for icol, dataset in enumerate(datasets):
            df = results_df[(results_df['target_column'] == target_column) & (results_df['dataset'] == dataset)]
            ax = axs[irow, icol] if nrows > 1 else axs[icol]
            
            # Ensure there is data to plot
            if df.empty:
                continue
            
            for i, model in zip(np.arange(0,(len(models))),models):
                # i=i+1
                i=i*2
                subset_df = df[df['model'] == model]
                j=0
                if subset_df.empty:
                    continue
                    
                for _,feature_set in enumerate(feature_sets):
                    feature_subset = subset_df[subset_df['feature_set'] == feature_set]
                    if feature_subset.empty:
                        continue
                    j=j+1
                    
                    # Calculate bar positions: now adjusted to group by model and feature set
                    bar_positions = i * (n_bars * bar_width) + j * bar_width 

                    test_mae_mean = feature_subset['test_mae_mean'].values
                    test_mae_std = feature_subset['test_mae_std'].values
                    train_mae_mean = feature_subset['train_mae_mean'].values
                    train_mae_std = feature_subset['train_mae_std'].values
                    
                    color = getcolor(feature_set)
                    hatch = gethatch(feature_set)
                    
                    # Plot bars for test MAE with error bars
                    ax.bar(bar_positions, 
                           test_mae_mean, 
                           width=bar_width, 
                           yerr=test_mae_std, 
                           capsize=5, 
                           color=color,
                           edgecolor='black',
                           hatch=hatch,)
                        #    label=feature_set if i == 0 else "")

                    ax.bar(bar_positions, 
                           train_mae_mean, 
                           width=bar_width, 
                           yerr=train_mae_std, 
                           capsize=5, 
                           edgecolor='black',
                           color='powderblue',
                           hatch=hatch,)
                           #label=feature_set if i == 0 else "")

                
                    
            ax.set_title(f'{target_column} - {dataset}')
            ax.set_xticks(indices*2 * n_bars * bar_width)
            ax.set_xticklabels(models, rotation=0,ha='left')
            ax.set_ylabel('CV-score (MAE)')
            # ax.legend(title='Feature Set')
            ax.grid()
    plt.suptitle(title,fontsize=18)
    plt.tight_layout()

    plt.show()

def plot_train_test_mae(results_df,test_column='test_mae_mean',train_column='train_mae_mean',figsize=(12, 10),title="Average MAE from Cross-Validation"):
    
    target_columns = np.unique(results_df['target_column'])
    datasets = np.unique(results_df['dataset'])
    nrows = len(target_columns)
    ncols = len(datasets)
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=False)
    total_group_width = 0.1
    feature_sets = results_df['feature_set'].drop_duplicates()
    models = results_df['model'].drop_duplicates()
    n_bars = len(feature_sets)
    
    bar_width = total_group_width / n_bars
    indices = np.arange(len(models))

    for irow, target_column in enumerate(target_columns):
        for icol, dataset in enumerate(datasets):
            df = results_df[(results_df['target_column'] == target_column) & (results_df['dataset'] == dataset)]
            ax = axs[irow, icol] if nrows > 1 else axs[icol]
            
            # Ensure there is data to plot
            if df.empty:
                continue
            
            for i, model in zip(np.arange(0,(len(models))),models):
                # i=i+1
                i=i*2
                subset_df = df[df['model'] == model]
                j=0
                if subset_df.empty:
                    continue
                    
                for _,feature_set in enumerate(feature_sets):
                    feature_subset = subset_df[subset_df['feature_set'] == feature_set]
                    if feature_subset.empty:
                        continue
                    j=j+1
                    
                    # Calculate bar positions: now adjusted to group by model and feature set
                    bar_positions = i * (n_bars * bar_width) + j * bar_width 

                    test_mae_mean = feature_subset['test_mae_mean'].values
                    test_mae_std = feature_subset['test_mae_std'].values
                    train_mae_mean = feature_subset['train_mae_mean'].values
                    train_mae_std = feature_subset['train_mae_std'].values
                    
                    color = getcolor(feature_set)
                    hatch = gethatch(feature_set)
                    
                    # Plot bars for test MAE with error bars
                    # ax.bar(bar_positions, 
                    #        test_mae_mean, 
                    #        width=bar_width, 
                    #        yerr=test_mae_std, 
                    #        capsize=5, 
                    #        color=color,
                    #        hatch=hatch,
                    #        label=feature_set if i == 0 else "")

                    ax.scatter(bar_positions,feature_subset[test_column].values,color='blue')
                    ax.scatter(bar_positions,feature_subset[train_column].values,color='red')
                    
            ax.set_title(f'{target_column} - {dataset}')
            ax.set_xticks(indices*2 * n_bars * bar_width)
            ax.set_xticklabels(models, rotation=0,ha='left')
            ax.set_ylabel('CV-score (MAE)')
            ax.legend(title='Feature Set')
            ax.grid()
    plt.suptitle(title,fontsize=18)
    plt.tight_layout()

    plt.show()

    
def plot_train_test_mae_score_and_ratio(results_df,operator='ratio',score1='test_mae_mean',score2='train_mae_mean',
                                        figsize=(12,10),title='',ylabel1='Train-Validate Ratio',
                                        ylabel2='CV-score (MAE)',alpha=1,ncol_legend=3,legend_loc = (.25, -.05),
                                        save_path=None):
    
    plt.rcParams.update({'font.size':14})
    target_columns = np.unique(results_df['target_column'])
    datasets = np.unique(results_df['dataset'])
    n_targets = len(target_columns)
    nrows=2
    ncols = len(datasets)
    
    renames = {'RandomForestRegressor':'RF','CatBoostRegressor':'CatBoost'}
    
    fig, axs = plt.subplots(nrows=nrows, ncols=n_targets*ncols, figsize=(20, 10), sharey=False)
    
    total_group_width = 10
    feature_sets = results_df['feature_set'].drop_duplicates()
    models = results_df['model'].drop_duplicates()
    n_bars = len(feature_sets)
    
    bar_width = (total_group_width*2) / n_bars
    indices = np.arange(len(models))
    k = 0
    for irow, target_column in enumerate(target_columns):
        for icol, dataset in enumerate(datasets):
            df = results_df[(results_df['target_column'] == target_column) & (results_df['dataset'] == dataset)]
            ax = axs[0,k]
            ax2 = axs[1,k]
            k = k+1

            # Ensure there is data to plot
            if df.empty:
                continue
            
            for i, model in zip(np.arange(0,(len(models))),models):
                # i=i+1
                i=i*2
                subset_df = df[df['model'] == model]
                if subset_df.empty:
                    continue
                j=0
                for _, feature_set in enumerate(feature_sets):
                    feature_subset = subset_df[subset_df['feature_set'] == feature_set]
                    if feature_subset.empty:
                        continue
                    j = j+1
                    
                    # Calculate bar positions: now adjusted to group by model and feature set
                    bar_positions = i * (n_bars * bar_width) + j * bar_width 

                    test_mae_mean = feature_subset['test_mae_mean'].values
                    test_mae_std = feature_subset['test_mae_std'].values
                    train_mae_mean = feature_subset['train_mae_mean'].values
                    train_mae_std = feature_subset['train_mae_std'].values
                    
                    # color = getcolor(feature_set)
                    # hatch = gethatch(feature_set)

                    if operator == 'ratio':
                        height =  feature_subset[score1].values/feature_subset[score2].values
                    elif operator == 'subtract':
                        height =  feature_subset[score1].values-feature_subset[score2].values
                    else:
                        ValueError(f'the operator {operator} is not defined') 

                    if 'mean' in score1:
                        yerr=test_mae_std
                    else:
                        yerr = None
                    # Plot bars for test MAE with error bars
                    ax.bar(bar_positions, 
                            height,
                           width=bar_width, 
                           capsize=5, 
                           edgecolor='black',
                        #    color=color,
                        #    hatch=hatch,
                           label=feature_set if i == 0 else "")
                    ax2.bar(bar_positions, 
                            feature_subset[score1],
                           width=bar_width, 
                           yerr=yerr,
                           capsize=5, 
                           edgecolor='black',
                        #    color=color,
                        #    hatch=hatch,
                           label=feature_set if i == 0 else "")
                    ax2.bar(bar_positions, 
                           feature_subset[score2],
                           width=bar_width, 
                        #    yerr=calculate_test_train_ratio_std(test_mae_mean,test_mae_std,train_mae_mean,train_mae_std), 
                        #    capsize=5, 
                           edgecolor='black',
                           color='powderblue',
                        #    hatch=hatch,
                           alpha=alpha,
                           label=feature_set if i == 0 else "")
            ax.grid()
            ax.set_title(f'{target_column} - {dataset}')
            ax.set_xticks(indices*2 * n_bars * bar_width + total_group_width+bar_width/2)
            ax.set_xticklabels([renames[m] for m in models.values], rotation=0,ha='center')
            ax.set_ylabel(ylabel1)
            # ax.legend(title='Feature Set')
            ax2.grid()
            ax2.set_title(f'{target_column} - {dataset}')
            ax2.set_xticks(indices*2 * n_bars * bar_width + total_group_width+bar_width/2)
            ax2.set_xticklabels([renames[m] for m in models.values], rotation=0,ha='center')
            ax2.set_ylabel(ylabel2)
            
    
    # Create a single legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title='Feature Set',
               bbox_to_anchor=legend_loc, 
               loc='center left', ncol=ncol_legend,)

    # plt.legend(title='Feature Set',ncols=2,bbox_to_anchor=(0,0))
    plt.suptitle(title,fontsize=18)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path,dpi=300)
        plt.show()

        
def get_defect_type_df(df):
    host_defects = df[['defects','base']]
    hosts = df['base'].unique()
    n_vacancy_list = []
    n_substitution_list = []
    impurity_set_list = []
    defect_val_list = []
    for host in hosts:
        defects = host_defects[host_defects['base']==host]
        for _,row_str in defects.iterrows():
            if type(row_str) == pd.Series:
                row = ast.literal_eval(row_str.iloc[0])
            else:
                row = row_str.values[0]
            # print(row)
            n_defects = len(row)
            n_vacancy = 0
            n_substitution = 0
            impurities = []
            row_vacancy = []
            defect_val = 0
            for defect in row:
                defect_type = defect['type']
                if defect_type == 'vacancy':
                    n_vacancy += 1
                    
                    el = Element(defect['element'])
                    defect_val = defect_val - el.atomic_mass
                    
                elif defect_type == 'substitution':
                    n_substitution += 1
                    impurities.append(defect['to'])
                    
                    from_el = Element(defect['from'])
                    to_el = Element(defect['to'])

                    defect_val = defect_val - from_el.atomic_mass + to_el.atomic_mass
                    
            impurity_set = str(set(impurities) if len(impurities)>0 else 'None')
            impurity_set_list.append(impurity_set)
            n_vacancy_list.append(n_vacancy)
            n_substitution_list.append(n_substitution)
            defect_val_list.append(defect_val)

    # print(hosts)
    # print(impurity_set_list)

    # df['n_vacancy'] = n_vacancy_list  
    # df['n_substitution'] = n_substitution_list  
    # df['impurity'] = impurity_set_list
    total_defect_list = [i+j for i,j in zip(n_vacancy_list,n_substitution_list)]
    df_copy = df.copy()
    df_copy = pd.concat([df_copy, pd.DataFrame({'n_vacancy': n_vacancy_list},index=df.index)], axis=1)
    df_copy = pd.concat([df_copy, pd.DataFrame({'n_substitution': n_substitution_list},index=df.index)], axis=1)
    df_copy = pd.concat([df_copy, pd.DataFrame({'impurity': impurity_set_list},index=df.index)], axis=1)
    df_copy = pd.concat([df_copy, pd.DataFrame({'total_defects': total_defect_list},index=df.index)], axis=1)
    df_copy = pd.concat([df_copy, pd.DataFrame({'defect_val': defect_val_list},index=df.index)], axis=1)

    # display(df_copy)
    return df_copy

def extract_distance_name(feature_set):
    pattern = r'(hellinger|tvd|emd)_'

    match = re.search(pattern=pattern,string=feature_set)
    if match:
        distance_name = match.group(1)
        return distance_name
    else:
        return None

class FeatureImportance:
    
    def __init__(self):
        pass
    
    def get_feature_importances(self,config,optimize=None,N=15,result_dirname='results_2'):
        config_info_df = pd.DataFrame({'database':[],'model':[],'target_column':[],'feature_set':[],'dataset':[]})
        main_df = pd.DataFrame()
        filename = 'feature_importance.csv'
        for db,model,target,feature_set,dataset in itertools.product(config['database'],config['model']['type'],config['target_column'],config['feature_set'],config['dataset']):
            feature_importances = loading_data.load_results(filename,db,feature_set,dataset,target,model,
                                                            optimize=optimize,result_dirname=result_dirname) 
            if N is not None:
                feature_importances = feature_importances.head(N)
            
            fi_1 = feature_importances.set_index('Feature').T.reset_index(drop=True)
            main_df = pd.concat([main_df,fi_1],ignore_index=True)


            config_info_row = pd.DataFrame({'database':[db],'model':[model],'target_column':[target],'feature_set':[feature_set],'dataset':[dataset]})
            config_info_df = pd.concat([config_info_df,config_info_row],ignore_index=True)

        main_df = pd.merge(config_info_df,main_df,left_index=True,right_index=True,how='outer').fillna(0)

        return main_df
    


    def sum_distribution_fi(self,df,feature_set):

        pass

        # # prefices = ['nn','ddf','mean_charge','adf1','adf2','rdf']
        # if feature_set == 'cfid' or feature_set == 'eo_chem_cfid' or feature_set == 'chem_dist0_cfid' or 'vpa' in feature_set:
        #     prefices = ['adf2','rdf','nn','adf1','ddf','mean_charge']
        #     sum_prefices_df = pd.DataFrame(index=df.index)
        #     prefix_features=[]
        #     for prefix in prefices:
        #         prefix_features = prefix_features + list( df.loc[:,df.columns.str.contains(f'_{prefix}_')].columns)
        #         sum_prefix_df = pd.DataFrame(df.loc[:,df.columns.str.contains(f'_{prefix}_')].sum(axis=1),columns=[prefix.upper()])
        #         sum_prefices_df = pd.merge(sum_prefices_df,sum_prefix_df,left_index=True,right_index=True)
        #     sum_other_df =pd.DataFrame( df.loc[:,~df.columns.isin(prefix_features)].sum(axis=1),columns=['others'])

        #     return pd.merge(sum_other_df,sum_prefices_df,left_index=True,right_index=True)
        # elif 'hellinger' in feature_set or 'tvd' in feature_set or 'emd' in feature_set:
            
        #     distance_name = extract_distance_name(feature_set)

        #     other_columns = pd.DataFrame(df.loc[:, ~df.columns.str.contains(distance_name)].sum(axis=1),columns=['others'])
        #     hellinger_columns = df.loc[:, df.columns.str.contains(distance_name)]
        #     # display(filtered_columns)
        #     fi_df = pd.merge(other_columns,hellinger_columns,left_index=True,right_index=True)

        #     return fi_df
        # else:
        #     pass
    

    def process_feature_importances(self,config,full_fi=False):
        # Get feature importances
        feature_importance_df = self.get_feature_importances(config, N=None)
        
        # Extract info_df to use as labels for each sum_fi_df
        info_df = feature_importance_df.iloc[:, :5]
        
        # Initialize an empty list to store all merged DataFrames
        merged_dfs = []
        
        # Iterate over each unique feature set
        for feature_set in np.unique(feature_importance_df.feature_set):
            feature_set_df = feature_importance_df[feature_importance_df['feature_set'] == feature_set].select_dtypes(include='float')
            # display(feature_set_df)
            sum_fi_df = self.sum_distribution_fi(feature_set_df, feature_set)

            if 'hellinger' in feature_set or 'tvd' in feature_set or 'emd' in feature_set:
                distance_name = extract_distance_name(feature_set)

                sum_fi_df.columns = [col.split(f'_{distance_name}')[0].upper() if f'_{distance_name}' in col else col for col in sum_fi_df.columns]
            
            merged_df = pd.merge(info_df, sum_fi_df, left_index=True,right_index=True)
            
            merged_dfs.append(merged_df)
        
        result_df = pd.concat(merged_dfs, axis=0)
        if full_fi:
            return result_df, feature_importance_df
        else:
            return result_df

            
            
class CompareDistribution:
    
    def __init__(self,df1,df2,hosts=['MoS2', 'InSe', 'P', 'GaSe', 'WSe2', 'BN']):
        self.df1 = df1
        self.df2 = df2
        self.hosts = hosts

    def split_cfid(self,**kwargs):
        self.cfid_data = {}
        for key in self.all_data.keys():
            self.cfid_data[key] =loading_data.split_cfid(self.all_data[key],**kwargs)
        
def get_Xy_from_result_dir(feature_set,target_column,info_df=None):
    X_train = loading_data.load_results('X_train.csv',database='2dmd',feature_set = feature_set,dataset='all_density',target_column=target_column,model='CatBoostRegressor',result_dirname='results_2')
    X_test = loading_data.load_results('X_test.csv',database='2dmd',feature_set = feature_set,dataset='all_density',target_column=target_column,model='CatBoostRegressor',result_dirname='results_2')

    y_train = loading_data.load_results('y_train.csv',database='2dmd',feature_set = feature_set,dataset='all_density',target_column=target_column,model='CatBoostRegressor',result_dirname='results_2')
    y_test = loading_data.load_results('y_test.csv',database='2dmd',feature_set = feature_set,dataset='all_density',target_column=target_column,model='CatBoostRegressor',result_dirname='results_2')
    X = pd.concat([X_train,X_test]).sort_index()
    if info_df is not None:
        X = pd.merge(X,info_df,left_index=True,right_index=True)
        X = get_defect_type_df(X)
    y = pd.concat([y_train,y_test]).sort_index()
    return X,y

from operator import itemgetter
import math
class GenDistribution():
    def __init__(self,structure):
        self.cutoff = 10.0
        self.intvl = 0.1
        self.max_n = 500
        self.binrng = np.arange(0,self.cutoff+self.intvl,self.intvl)
        self.structure = structure

    def _calculate_all_distances(self):
        self.neighbors_lst = self.structure.get_all_neighbors(self.cutoff)
        mapper = map(lambda x: [itemgetter(1)(e) for e in x], self.neighbors_lst)
        self.all_distances = np.concatenate(tuple(mapper))

    def _get_hist_bins(self):
        self.dist_hist, self.dist_bins = np.histogram(self.all_distances,bins=self.binrng,density=False)

    def _get_rdf(self):
        self.shell_vol = 4.0 / 3.0 * math.pi * (np.power(self.dist_bins[1:], 3) - np.power(self.dist_bins[:-1], 3))
        number_density = len(self.structure) / self.structure.volume
        self.denominator = self.shell_vol * number_density * len(self.neighbors_lst)
        raw_rdf = self.dist_hist / self.shell_vol / number_density / len(self.neighbors_lst)
        self.rdf = [round(i,4) for i in raw_rdf]

    def _get_nn(self):
        self.nn = self.dist_hist/float(len(self.structure))

    def _get_bins(self):
        self.bins = self.dist_bins[:-1]
        
        
    def get_rdf_nn(self):
        self._calculate_all_distances()
        self._get_hist_bins()
        self._get_rdf()
        self._get_nn()
        self._get_bins()
        return self.rdf,self.nn,self.bins
    

    
    def nbor_list(self, rcut=10.0, c_size=12.0):
        """Generate neighbor info."""
        max_n = self.max_n
        nbor_info = {}
        struct_info = self.get_structure_data(c_size)
        coords = np.array(struct_info["coords"])
        nat = struct_info["nat"]
        new_symbs = struct_info["new_symbs"]
        lat = np.array(struct_info["lat"])
        different_bond = {}
        nn = np.zeros((nat), dtype="int")
        # print ('max_n, nat',max_n, nat)
        dist = np.zeros((max_n, nat))
        nn_id = np.zeros((max_n, nat), dtype="int")
        bondx = np.zeros((max_n, nat))
        bondy = np.zeros((max_n, nat))
        bondz = np.zeros((max_n, nat))
        for i in range(nat):
            for j in range(i + 1, nat):
                diff = coords[i] - coords[j]
                ind = np.where(np.fabs(diff) > np.array([0.5, 0.5, 0.5]))
                diff[ind] -= np.sign(diff[ind])
                new_diff = np.dot(diff, lat)
                dd = np.linalg.norm(new_diff)
                if dd < rcut and dd >= 0.1:
                    sumb_i = new_symbs[i]
                    sumb_j = new_symbs[j]
                    comb = "_".join(
                        sorted(str(sumb_i + "_" + sumb_j).split("_"))
                    )
                    different_bond.setdefault(comb, []).append(dd)

                    # print ('dd',dd)
                    nn_index = nn[i]  # index of the neighbor
                    nn_index1 = nn[j]  # index of the neighbor
                    if nn_index < max_n and nn_index1 < max_n:
                        nn[i] = nn[i] + 1
                        dist[nn_index][i] = dd  # nn_index counter id
                        nn_id[nn_index][i] = j  # exact id
                        bondx[nn_index][i] = new_diff[0]
                        bondy[nn_index][i] = new_diff[1]
                        bondz[nn_index][i] = new_diff[2]
                        nn[j] = nn[j] + 1
                        dist[nn_index1][j] = dd  # nn_index counter id
                        nn_id[nn_index1][j] = i  # exact id
                        bondx[nn_index1][j] = -new_diff[0]
                        bondy[nn_index1][j] = -new_diff[1]
                        bondz[nn_index1][j] = -new_diff[2]
                    else:
                        self.nb_warn = (
                            "Very large nearest neighbors observed "
                            + str(nn_index)
                        )
        nbor_info["dist"] = dist
        nbor_info["nat"] = nat
        nbor_info["nn_id"] = nn_id
        nbor_info["nn"] = nn
        nbor_info["bondx"] = bondx
        nbor_info["bondy"] = bondy
        nbor_info["bondz"] = bondz
        nbor_info["different_bond"] = different_bond
        # print ('nat',nat)

        return nbor_info

    def _get_dist_cutoffs(self):
        max_cut = 10.0
        arr = []
        y,z,x = self.get_rdf_nn()
        for i, j in zip(x,z):
            if j > 0.0:
                arr.append(i)
        rcut_buffer = 0.11
        io1,io2,io3 = 0,1,2
        try:
            delta = arr[io2] - arr[io1]
            while delta < rcut_buffer and arr[io2] < max_cut:
                io1 = io1 + 1
                io2 = io2 + 1
                io3 = io3 + 1
                delta = arr[io2] - arr[io1]
            rcut1 = (arr[io2] + arr[io1]) / float(2.0)
        except Exception:
            print("Warning:Setting first nbr cut-off as minimum bond-angle")
            rcut1 = arr[0]
            pass
        try:
            delta = arr[io3] - arr[io2]
            while (
                delta < rcut_buffer
                and arr[io3] < max_cut
                and arr[io2] < max_cut
            ):
                io2 = io2 + 1
                io3 = io3 + 1
                delta = arr[io3] - arr[io2]
            rcut2 = float(arr[io3] + arr[io2]) / float(2.0)
        except Exception:
            print("Warning:Setting first and second nbr cut-off equal")
            print("You might consider increasing max_n parameter")
            rcut2 = rcut1
            pass
        # rcut, rcut_dihed = get_prdf(s=s)
        # rcut_dihed=min(rcut_dihed,max_dihed)
        return rcut1, rcut2
    def ang_dist(self, nbor_info={}, plot=False):
        """
        Get  angular distribution function upto first neighbor.

        Args:
            struct_info: struct information

            max_n: maximum number of neigbors

            c_size: max. cell size

            plot: whether to plot distributions


        Retruns:

            ang_hist1: Angular distribution upto first cut-off

            ang_bins1: angle bins
        """
        # rcut1,rcut2=self.get_dist_cutoffs()
        # rcut=rcut1
        # nbor_info=self.nbor_list()
        znm = 0
        nat = nbor_info["nat"]
        dist = nbor_info["dist"]
        # nn_id = nbor_info["nn_id"]
        nn = nbor_info["nn"]
        bondx = nbor_info["bondx"]
        bondy = nbor_info["bondy"]
        bondz = nbor_info["bondz"]

        ang_at = {}

        for i in range(nat):
            for in1 in range(nn[i]):
                # j1 = nn_id[in1][i]
                for in2 in range(in1 + 1, nn[i]):
                    # j2 = nn_id[in2][i]
                    nm = dist[in1][i] * dist[in2][i]
                    if nm != 0:
                        rrx = bondx[in1][i] * bondx[in2][i]
                        rry = bondy[in1][i] * bondy[in2][i]
                        rrz = bondz[in1][i] * bondz[in2][i]
                        cos = float(rrx + rry + rrz) / float(nm)
                        if cos <= -1.0:
                            cos = cos + 0.000001
                        if cos >= 1.0:
                            cos = cos - 0.000001
                        deg = math.degrees(math.acos(cos))
                        ang_at.setdefault(round(deg, 3), []).append(i)
                    else:
                        znm = znm + 1
        angs = np.array([float(i) for i in ang_at.keys()])
        norm = np.array(
            [float(len(i)) / float(len(set(i))) for i in ang_at.values()]
        )
        ang_hist, ang_bins = np.histogram(
            angs, weights=norm, bins=np.arange(1, 181.0, 1), density=False,
        )
        # if plot == True:
        #    plt.bar(ang_bins[:-1], ang_hist)
        #    plt.savefig("ang1.png")
        #    plt.close()

        return ang_hist, ang_bins


def get_info_df(incl_full_df = False):
    full_high_df = loading_data.full_data('2dmd','all_high_density','cfid')
    full_high_df['density'] = 'high'
    full_low_df = loading_data.full_data('2dmd','all_low_density','cfid')
    full_low_df['density'] = 'low'
    full_df = pd.concat([full_high_df,full_low_df])
    full_df = get_defect_type_df(full_df)
    info_df = full_df[['base','density','defects','description'
                       ,'n_vacancy','n_substitution','total_defects',
                       'formation_energy_per_site','homo_lumo_gap_min']]
    if incl_full_df == True:
        return info_df, full_df
    else:
        return info_df

     
class DistanceAnalysis:
    def __init__(self,dist_dict,info_df):
        self.info_df = info_df
        self.dist_dict = dist_dict
        # self.filter_df(filter_dict)
        # self._assign_label_to_filt_df()
        
    
    def _get_filter_index(self,X,filter_dict):
        """ Return the filtered index using `filter_dict` as a filter

        Args:
            X (pandas.DataFrame): DataFrame to filter
            filter_dict (dictionary): {column_name: [values]} 

        Returns:
            list: 
        """
        filt_X = X[X[list(filter_dict)].isin(filter_dict).all(axis=1)]
    
        return filt_X.index.tolist() 
    
 
    def _non_outlier_df(self,df,col_name):
        col = df[col_name]
        des_col = col.describe()
        one_third = des_col['25%']
        two_third = des_col['75%']

        return df[(df[col_name]<=two_third)]
                
    def filter_df(self,filter_dict):
        filt_index = self._get_filter_index(self.info_df,filter_dict)
        self.filt_index = self.info_df.loc[filt_index].index.to_list()

    def _assign_label_to_filt_df(self,X,defect_type=False):

        self.filt_df = pd.merge(X,self.info_df.loc[self.filt_index],left_index=True,right_index=True)
        ## for low density only
        if defect_type:
            type_label = self.info_df.loc[self.filt_index,'description']
            alphabet_label = [k[0] for k in type_label.values]
            number_label = [k[-1] for k in type_label.values]
            self.filt_df['alphabet_label'] = alphabet_label
            self.filt_df['number_label'] = number_label
        ##########################
    
    def plot(self,plot_function, **kwargs):
        plot_function(self.filt_df,**kwargs)

    # def plot_all_distances(self,info_df,filter_dict,distribution,axs=None):
    #     if axs is None:
    #         fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(12,4))
    #     for distance, X in self.dist_dict.items():
    #         self.filter_df(filter_dict)
    #         self._assign_label_to_filt_df(,defect_type=True)


if __name__ == "__main__":
    notebooks_path = os.path.join(this_file_path,'..')
    print(this_file_path)
    print(notebooks_path)
    

        

        
    
    
        



                
        