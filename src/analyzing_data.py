import yaml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
# from src.utils import load_configs

distance_pattern = {'euclidean_cfid':'o',
                    'euclidean_pristine_cfid':'o',
                    'hellinger_cfid':'|',
                    'hellinger_pristine_cfid':'|',
                    'pca_cfid':'x',
                    'kernelpca_cfid':'x',
                    'cfid':'*',
                    'kazeev':'-',
                    'reduced_kazeev':'-'}

def my_unique(LIST):
    new_list = []
    for i in LIST:
       if i in new_list:
           continue
       else:
           new_list.append(i) 
    return new_list
def sort_df(df):
    # Define the desired order for the 'feature_set' column
    desired_order = ['euclidean_cfid','euclidean_pristine_cfid', 'hellinger_cfid','hellinger_pristine_cfid', 'cfid', 'pca_cfid', 'kernelpca_cfid']

    # Convert 'feature_set' column to a categorical type with the specified order
    df['feature_set'] = pd.Categorical(df['feature_set'], categories=desired_order, ordered=True)

    # Sort the DataFrame by the 'feature_set' column
    df_sorted = df.sort_values(by='feature_set') 
    return df_sorted
        

def get_time_for_5f_cv(result_dir):

    f = open(result_dir,'r')
    file = f.read()
    if float(file.split(" ")[-1]) < 0:
        return -float(file.split(" ")[-1])
    else:
        return float(file.split(" ")[-1])

def plot_cv_time(df,SAVEPATH=None):
    df_pivot = df.pivot(index='dataset',columns='feature_set',values='cv_time')
    df_pivot.plot(kind='bar',figsize=(12, 6))
    plt.xticks(rotation=0) 
    plt.legend(title='Distance Measure')
    plt.xlabel('Dataset')
    plt.ylabel('time (second)')
    plt.title('Time taken for performing 5-fold-CV')
    plt.tight_layout()
    if SAVEPATH is not None:
        plt.savefig(SAVEPATH,dpi=300)
        plt.close()

    else:
        plt.show()

def plot_test_mae_bar_chart(df,SAVEPATH=None):
    df_pivot = df.pivot(index='dataset',columns='feature_set',values='test_mae')
    df_pivot.plot(kind='bar')
    plt.xticks(rotation=45)
    plt.legend(title='Distance Measure')
    plt.xlabel('Dataset')
    plt.ylabel('MAE')
    plt.title('Blind Test Set')
    plt.tight_layout()
    if SAVEPATH is not None:
        plt.savefig(SAVEPATH,dpi=300)
        plt.close()

    else:
        plt.show()


def plot_mean_test_mae_bar_chart(df:pd.DataFrame(),config=None,SAVEPATH=None,except_config=[]):
    df_pivot = df.pivot_table(index='dataset',columns='feature_set',values=['test_mae_mean','test_mae_std'])
    total_group_width = 0.8

    n_bars = len(df['feature_set'].drop_duplicates())
    n_groups = len(df_pivot.index)
    
    bar_width = total_group_width/n_bars
    indices = np.arange(n_groups)
    
    plt.figure(figsize=(10,6))

    count = 0
    for feature_set in my_unique(df['feature_set']):
        if feature_set in except_config:
            continue
        count += 1
        bar_position = indices - (total_group_width/2) + (count*bar_width) + (bar_width/2)
        plt.bar(x=bar_position,height=df_pivot['test_mae_mean'][feature_set],width = bar_width,yerr= df_pivot['test_mae_std'][feature_set],capsize=5,label=feature_set,hatch = distance_pattern[feature_set])
    plt.xlabel('Dataset')
    plt.ylabel('MAE')
    plt.title('Average MAE from 5-fold CV')
    plt.legend()
    plt.xticks(indices,labels=df_pivot.index)#,rotation=45)
    plt.tight_layout()
    if SAVEPATH is not None:
        plt.savefig(SAVEPATH,dpi=300)
        plt.close()

    else:
        plt.show()
    

def plot_separate_mean_test_mae_bar_chart(df:pd.DataFrame(),target,model,config_colors,config_labels,SAVEPATH=None,except_config=[],figsize=(16, 4)):

    df = df[df['model']==model]
    
    datasets = np.unique(df['dataset'])
    datasets = [fs for fs in datasets if fs != 'wse2_low_density']
    
    target_labels = {'homo_lumo_gap_min': 'HOMO-LUMO gap minimum','formation_energy': 'Formation Energy'}
    
    dataset_labels = {
                      'all_low_density':'low density defects',
                      'all_high_density':'high density defects',
                      'mos2_low_density': '$MoS_2$ low density defects'
                      }
    


    fig, axs = plt.subplots(1, len(datasets), figsize=figsize)

    for i, dataset in enumerate(datasets):
        ax = axs[i]
        dataset_data = df[df['dataset'] == dataset]
        mean_mae = dataset_data['test_mae_mean'].values
        std_mae = dataset_data['test_mae_std'].values
        feature_set = dataset_data['feature_set'].values
        
        colors = [config_colors['feature_set'][fs] for fs in feature_set]
        labels = [config_labels['feature_set'][fs] for fs in feature_set]
        
        ax.bar(np.arange(len(feature_set)), mean_mae, yerr=std_mae, capsize=5, color=colors)
        ax.set_title(dataset_labels[dataset])
        ax.set_xticks(np.arange(len(labels)),labels, rotation=45,ha='right')
        ax.grid()
        
    axs[0].set_ylabel('MAE')
    fig.suptitle(f'Average MAE from 5-fold CV - {target_labels[target]}', fontsize=16)
    plt.tight_layout()
    if SAVEPATH is None:
        plt.show()
    else:
        plt.savefig(SAVEPATH,dpi=300)
        plt.close()


        
def plot_test_trian_ratio_bar_chart(df:pd.DataFrame(),config,SAVEPATH=None,except_config=[]):
    df_pivot = df.pivot(index='dataset',columns='feature_set',values=['test_train_ratio','test_train_ratio_std'])
    total_group_width = 0.8

    n_bars = len(df['feature_set'].drop_duplicates())
    n_groups = len(df_pivot.index)
    
    bar_width = total_group_width/n_bars
    indices = np.arange(n_groups)
    
    plt.figure(figsize=(10,6))

    count = 0
  
    for feature_set in my_unique(df['feature_set']):
        if feature_set in except_config:
            continue
        count += 1
        bar_position = indices - (total_group_width/2) + (count*bar_width) + (bar_width/2)
        plt.bar(x=bar_position,height=df_pivot['test_train_ratio'][feature_set],width = bar_width,yerr= df_pivot['test_train_ratio_std'][feature_set],capsize=5,label=feature_set,hatch=distance_pattern[feature_set])
    plt.xlabel('Dataset')
    plt.ylabel('Test Train Ratio')
    plt.title('Test Train Ratio by Distance Measure')
    plt.legend()
    plt.xticks(indices,labels=df_pivot.index)#,rotation=45)
    plt.tight_layout()
    if SAVEPATH is not None:
        plt.savefig(SAVEPATH,dpi=300)
        plt.close()

    else:
        plt.show()

    
def get_blind_test_results(result_dir):

    eval_results_name = os.path.join(result_dir,'eval_metrics.csv')
    eval_results= pd.read_csv(eval_results_name)
    test_mae = eval_results['MAE_test'].iloc[0]
    train_mae = eval_results['MAE_train'].iloc[0]
    return train_mae,test_mae

def merge_configs(config1,config2):
    # Initialize the resulting dictionary
    C = {}

    for key in config1:
        if isinstance(config1[key], list) and isinstance(config2[key], list):
            # Merge lists and remove duplicates
            C[key] = list(set(config1[key] + config2[key]))
        elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
            # Recursively merge nested dictionaries
            C[key] = merge_configs(config1[key], config2[key])
        else:
            # If the key is not a list or dict, just take the value from A (assuming A and B have the same non-list, non-dict values for the same keys)
            C[key] = config1[key]
    
    return C


def load_config_file(config_path):
    
    with open(config_path,'r') as f:
        config = yaml.safe_load(f)
    
    return config

def calculate_test_train_ratio_std(test,std_test,train,std_train):
    F = test/train
    return F*(std_test/test * std_train/train)
    

