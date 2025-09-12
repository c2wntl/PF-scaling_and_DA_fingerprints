from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

def bar_compare_train_test(cv_score,SAVEPATH=None,TITLE=None,bar_width=0.35,best_score=False):
    """_summary_
        Plotting the comparison of the mean value of R2 and MAE of CV results
    Args:
        cv_score (_type_): _description_
        SAVEPATH (_type_, optional): _description_. Defaults to None.
        TITLE (_type_, optional): _description_. Defaults to None.
        bar_width (float, optional): _description_. Defaults to 0.35.
    """

    if best_score:
        # This condition is for 
        test_means = [cv_score.loc[:,'test_r2_mean'].values[0], cv_score.loc[:,'test_mae_mean'].values[0]]
        train_means = [cv_score.loc[:,'train_r2_mean'].values[0], cv_score.loc[:,'train_mae_mean'].values[0]]

        test_stds = [cv_score['test_r2_std'], cv_score['test_mae_std']]
        train_stds = [cv_score['train_r2_std'], cv_score['train_mae_std']]
        print(test_means)

    else:
        
        # Mean values
        test_means = [cv_score['test_r2'].mean(), -cv_score['test_neg_mean_absolute_error'].mean()]
        train_means = [cv_score['train_r2'].mean(), -cv_score['train_neg_mean_absolute_error'].mean()]

        # Standard deviations
        test_stds = [cv_score['test_r2'].std(), cv_score['test_neg_mean_absolute_error'].std()]
        train_stds = [cv_score['train_r2'].std(), cv_score['train_neg_mean_absolute_error'].std()]

    # Labels
    metrics = ['R2', 'MAE']

    # Set the width of the bars
    bar_width = 0.35

    # Set the x locations for the groups
    index = np.arange(len(metrics))

    # Plot bars
    fig, ax = plt.subplots()
    bars_test = ax.bar(index - bar_width/2, test_means, bar_width, label='Test', yerr=test_stds, capsize=5)
    bars_train = ax.bar(index + bar_width/2, train_means, bar_width, label='Train', yerr=train_stds, capsize=5)

    # Add labels, title, and legend
    if  test_means[1] <= 1:
        ax.set_ylim(0,1.1)
    ax.axhline(y=0,alpha=0.7,color='grey',ls='--')
    ax.grid()
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Mean Values')
    ax.set_title(TITLE)
    ax.set_xticks(index)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Annotate the bars with their values
    for i, bar in enumerate(bars_test):
        plt.text(bar.get_x() + bar.get_width() / 2, 0,#bar.get_height() - 0.05, 
                f'{test_means[i]:.4f}', ha='center', va='bottom')

    for i, bar in enumerate(bars_train):
        plt.text(bar.get_x() + bar.get_width() / 2, 0,#bar.get_height() - 0.05, 
                f'{train_means[i]:.4f}', ha='center', va='bottom')

    if SAVEPATH is not None:
        # check_parent_dir(SAVEPATH)
        plt.savefig(SAVEPATH,bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
        
    

def parity(ypred,ytrue,TITLE=None,XLABEL=None,YLABEL=None,SAVEPATH=None):
    plt.scatter(ypred,ytrue,color='blue',alpha=0.5)
    plt.plot([min(ypred),max(ypred)],[min(ytrue),max(ytrue)],color = 'red',ls='--')
    if TITLE is not None:
        plt.title(TITLE)

    if YLABEL is not None:
        plt.ylabel(YLABEL)
    else:
        plt.ylabel('Predicted Values')

    if XLABEL is not None:
        plt.xlabel(XLABEL)
    else:
        plt.xlabel('True Values')

    if SAVEPATH is not None:
        # check_parent_dir(SAVEPATH)
        plt.savefig(SAVEPATH,dpi=300)
        plt.close()
    else:
        plt.show()
    
def feature_importance(model,features,N=15,SAVEPATH=None):
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    plt.figure()
    # plt.barh(feature_importance_df['Feature'].head(N).sort_values(), feature_importance_df['Importance'].head(N).sort_values())
    plt.barh(feature_importance_df['Feature'].head(N), feature_importance_df['Importance'].head(N))
    plt.title('Feature Importances')
    plt.tight_layout()
    if SAVEPATH is not None:
        # check_parent_dir(SAVEPATH)
        plt.savefig(SAVEPATH,dpi=300)
        plt.close()
    else:
        plt.show()  
        