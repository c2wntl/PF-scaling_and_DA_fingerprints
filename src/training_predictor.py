from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import cross_validate
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso

import time
import pandas as pd
import numpy as np


from sklearn.metrics import make_scorer

def weighted_mae(y_true,y_pred,weights):
    weights = weights.loc[y_true.index]
    
    if isinstance(weights, pd.DataFrame):
        weight_values = weights.iloc[:, 0].values
    elif isinstance(weights, pd.Series):
        weight_values = weights.values
        
    return -np.average(np.abs(y_true - y_pred), weights=weight_values)

def weighted_mae_scorer(weights):
    return make_scorer(lambda y_true, y_pred: weighted_mae(y_true, y_pred, weights))

def initialize_model(model_name, params=None, n_jobs=2,loss_function='mae'):
    params = params or {} 
    loss_function_params = {'mae':'absolute_error',
                            'mse':'squared_error'}
    if model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(
            random_state=42,
            criterion=loss_function_params[loss_function],
            n_jobs=n_jobs,
            **params
        )
        
    elif model_name == 'CatBoostRegressor':
        loss_function_params = {'mae':'MAE',
                                'mse':'RMSE'}
        model = CatBoostRegressor(
            iterations=1000,
            verbose=0,
            loss_function=loss_function_params[loss_function],
            thread_count=n_jobs,
            **params
        )

    elif model_name == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor(criterion=loss_function_params[loss_function],
                                    random_state=42,)

    elif model_name == 'ExtraTreesRegressor':
        model = ExtraTreesRegressor(criterion=loss_function_params[loss_function],
                                    random_state=42,n_jobs=n_jobs)
    elif model_name == 'AdaBoostRegressor':
        model = AdaBoostRegressor(random_state=42,)
    elif model_name == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor(random_state = 42, criterion=loss_function_params[loss_function])
    elif model_name =='Ridge':
        model = Ridge(random_state=42)
    elif model_name == 'Lasso':
        model = Lasso(random_state=42)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model

def performing_cv(kfold,model,X_train,y_train,scoring = None,n_jobs=None,weights = None):
    startTime = time.time()

    if scoring is None:
        scoring = {
           'r2' : 'r2',
           'neg_mean_absolute_error': 'neg_mean_absolute_error'
        }
        
    if weights is not None:
        scoring['weighted_mae'] = weighted_mae_scorer(weights)

    if isinstance(model, CatBoostRegressor) and weights is not None:
        cv_score = cross_validate(
            model, X_train, y_train, cv=kfold, scoring=scoring,
            return_train_score=True, n_jobs=n_jobs,
        )
    else:
        cv_score = cross_validate(
            model, X_train, y_train, cv=kfold, scoring=scoring,
            return_train_score=True, n_jobs=n_jobs
        )
    endTime = time.time()

    return cv_score, endTime-startTime

        
def random_search(kfold,model, param_grid, X, y,scoring = None,n_iter=1,weights=None,n_jobs=1):
    if scoring is None:
        scoring = {
            'r2' : 'r2',
            'neg_mean_absolute_error': 'neg_mean_absolute_error'
            }

    if weights is not None:
        scoring['weighted_mae'] = weighted_mae_scorer(weights)
        refit ='weighted_mae'
    else:
        refit = 'neg_mean_absolute_error'
        
    search = RandomizedSearchCV(
                                model, param_grid, n_iter=n_iter, cv=kfold, 
                                verbose=5, n_jobs=n_jobs,
                                return_train_score=True,random_state=42,
                                scoring=scoring, refit=refit
                                )


    search.fit(X, y)
    results_df = pd.DataFrame(search.cv_results_)
    best_index = search.best_index_
    best_train_score = results_df.loc[best_index, :]
    return search.best_params_, results_df,best_train_score


def grid_search(kfold,model, param_grid, X, y,scoring=('neg_mean_absolute_error','r2'),n_jobs=1,weights=None):
    scoring = {
           'r2' : 'r2',
           'neg_mean_absolute_error': 'neg_mean_absolute_error'
        }
    if weights is not None:
        scoring['weighted_mae'] = weighted_mae_scorer(weights)
        refit = 'weighted_mae'
    else:
        refit = 'neg_mean_absolute_error'
        
    search = GridSearchCV(model, param_grid, cv=kfold, verbose=5, n_jobs=n_jobs,
                               scoring=scoring,return_train_score=True,
                                refit=refit
                               )
    search.fit(X, y)
    results_df =  pd.DataFrame(search.cv_results_)
    best_index = search.best_index_
    best_train_score = results_df.loc[best_index,:]
    return search.best_params_,results_df,best_train_score