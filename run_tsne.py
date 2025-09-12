import os,sys
import pandas as pd

from src import loading_data
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import colorcet as cc
import json

def get_tsne(X,selected_index=None,perp=30):
    if selected_index is not None:
        X = X.loc[selected_index, :]
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)

    X_embedded = tsne.fit_transform(X_normalized)
    df_tsne = pd.DataFrame(X_embedded, columns=['TSNE-1', 'TSNE-2'], index=X.index)
    return df_tsne

def plot_tsne(df_tsne, label, selected_index=None,is_label=True,title='t-SNE Visualization '):
    if selected_index is not None:
        df_tsne = df_tsne.loc[selected_index, :]
        label = label.loc[selected_index]
    plt.rcParams.update({'font.size':14})
    # Create beautiful plot
    figure = plt.figure(figsize=(10, 8))
    palette = cc.glasbey[:len(label.unique())]  # Distinct colors
    df_tsne = pd.merge(df_tsne, label, left_index=True, right_index=True)
    # display(df_tsne)
    sns.scatterplot(
        data=df_tsne,
        x='TSNE-1', y='TSNE-2',
        hue=label.name,
        palette=palette,
        alpha=0.7,
        s=100,               # Larger markers
        edgecolor='black',   # Marker edge
        linewidth=0.2,
        legend=is_label
    )
    if is_label:
        plt.legend(title="hosts", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18, title_fontsize=20,)
                #    frameon= True)
    figure.subplots_adjust(right=0.75)
    # Style and aesthetics
    plt.xlabel("TSNE-1", fontsize=18) #, fontweight='bold')
    plt.ylabel("TSNE-2", fontsize=18) #, fontweight='bold')
    # plt.axis('off')
    plt.title(title, fontsize=22,fontweight='bold', pad=20)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box')

    sns.despine()  # Remove top and right axes for a cleaner look
    # plt.tight_layout()

    # plt.show() 
    return figure

def plot_one_low_host_tsne(df_tsne, base ,title='t-SNE Visualization with Colors & Markers',is_label=True,
                           ):
    fig_width = 10
    plt.rcParams.update({'font.size':14})

    unique_labels = base.unique()
    num_labels = len(unique_labels)
    color_palette = cc.glasbey[:num_labels]  # Unique colors

    marker_styles = ['o', 's', 'D', 'P', 'X', '^', 'v', '<', '>']

    figure = plt.figure(figsize=(fig_width, 8))
    
    for i, label in enumerate(unique_labels):
        marker = marker_styles[i % len(marker_styles)]
        color = color_palette[i]

        subset = df_tsne.loc[base[base == label].index]  # Fix: Align base index

        plt.scatter(
            subset['TSNE-1'], subset['TSNE-2'],
            label=label, color=color, marker=marker,
            s=80,
            edgecolors='k', 
            linewidth=0.2,
            alpha=0.4
        )

    if is_label:
        plt.legend(title="defect types", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=14)
    # Style and aesthetics
    figure.subplots_adjust(right=0.75)
    plt.xlabel("TSNE-1", fontsize=18)#, fontweight='bold')
    plt.ylabel("TSNE-2", fontsize=18)#, fontweight='bold')
    plt.title(title, fontsize=20, fontweight='bold', pad=20)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    sns.despine()  # Remove top and right axes for a cleaner look
    return figure

def get_savepath(feature_set):
    savepath = os.path.join('results_2_tsne',feature_set)
    os.makedirs(os.path.join('results_2_tsne',feature_set),exist_ok=True)
    return savepath

def get_host_density():
    with open('dataset/labels/2dmd_host_density.json','r') as f:
        index_dict = json.load(f)
    return index_dict


def filt_host(df,host,density,index_dict):

    host_density_index = set(index_dict[host][density])
    df_index = set(df.index)

    host_df_index = df_index.intersection(host_density_index)

    return df.loc[list(host_df_index),:]

if __name__ == '__main__':
    configs = loading_data.load_config_file('configs/r4_config_test.yaml')
    target = 'formation_energy_per_site' # this is arbitary
    feature_sets = configs['feature_set']

    n_feature_sets = len(feature_sets)

    host_index_dict = get_host_density()

    for i,feature_set in enumerate(feature_sets):#enumerate(feature_sets):
        print(f'>>>>>>>>>>> {i+1}/{n_feature_sets}: {feature_set}')
        X_train = loading_data.load_results('X_train.csv',dataset='all_density',database='2dmd',
                              feature_set=feature_set,target_column=target,
                              model='CatBoostRegressor',result_dirname='results_2')
        X_test = loading_data.load_results('X_test.csv',dataset='all_density',database='2dmd',
                              feature_set=feature_set,target_column=target,
                              model='CatBoostRegressor',result_dirname='results_2')

                              
        X = pd.concat([X_train,X_test])
        save_path = get_savepath(feature_set)
        save_path = os.path.join(save_path,'entire_dataset')
        os.makedirs(save_path,exist_ok=True)

        wse2_low_index = host_index_dict['WSe2']['low']
        mos2_low_index = host_index_dict['MoS2']['low']

        X_wse2_low = filt_host(X,'WSe2','low',index_dict=host_index_dict)
        X_mos2_low = filt_host(X,'MoS2','low',index_dict=host_index_dict)

        all_host_tsne = get_tsne(X)
        wse2_low_tsne = get_tsne(X_wse2_low)
        mos2_low_tsne = get_tsne(X_mos2_low)

        all_host_tsne.to_csv(os.path.join(save_path,'all_host.csv'))
        print(all_host_tsne.head())
        wse2_low_tsne.to_csv(os.path.join(save_path,'wse2_low.csv'))
        mos2_low_tsne.to_csv(os.path.join(save_path,'mos2_low.csv'))


        




