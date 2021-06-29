import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle
import time
import warnings

res_path = "/home/clairegayral/Documents/openclassroom/res/P3/"


def get_dict_index(res):
    '''
    Get the index of different values in the results_cv dictionnary of GridSearchCV
    '''
    dict_index_values = {}
    for param_name in res["params"][0].keys() : 
        index_values = []
        param_values = np.unique([dict_params[param_name] for dict_params in res["params"] ])
        for value in param_values :
            index_value = [dict_params[param_name] == value for dict_params in res["params"]]
            index_values.append(index_value)
        dict_index_values[param_name] = index_values
    return(dict_index_values)

def get_score_and_time(param_name, res, dict_index_values):
    '''
    Get mean score and the mean time for the parameter param_name, from res and dict_index_values

    Parameters:
    -----------------------------------------
    param_name : str of the parameter name
    res : results_cv dictionnary of GridSearchCV
    dict_index_values : result from get_dict_index(res)
    
    Returns:
    -----------------------------------------
    mean score, mean time    
    '''
    mean_score_param = []
    time_param = []
    for index_value in dict_index_values[param_name] :
        mean_score_param.append(np.mean(res["mean_test_score"][index_value]))
        time_tmp = (res["mean_fit_time"] + res["mean_score_time"])[index_value]
        time_param.append(np.mean(time_tmp))
    return(mean_score_param,time_param)

def cv_plot_score_and_time(x, y_score, y_time, subplot = [1,2,1],
                           param_name="", x_log_scale = True):
    '''

    Parameters:
    -----------------------------------------

    
    Returns:
    -----------------------------------------
    '''
    subplot = subplot.copy()
    plt.subplot(*subplot)
    if x_log_scale : 
        plt.xscale("log")
    plt.plot(x, y_score)
    plt.xlabel("hyperparam. " + param_name)#get param name 
    plt.ylabel("score")
    plt.title("Score in Cross validation")
    subplot[2] += 1
    plt.subplot(*subplot)
    if x_log_scale : 
        plt.xscale("log")
    plt.plot(x, y_time)
    plt.xlabel("hyperparam. " + param_name)#get param name 
    plt.ylabel("time (s)")
    plt.title("Time in Cross validation")

def plot_cv_res(res,dict_log_param) :
    '''

    Parameters:
    -----------------------------------------

    
    Returns:
    -----------------------------------------
    '''
    dict_index_values = get_dict_index(res)
    nb_params = len(res["params"][0].keys())
    plt.figure(figsize=(12,5*nb_params))
    subplot_pos = [nb_params,2,1]

    for param_name in res["params"][0].keys() :
        x = np.unique([dict_params[param_name] for dict_params in res["params"]])
        y_score, y_time = get_score_and_time(param_name, res, dict_index_values)
        log_flag = dict_log_param[param_name]
        cv_plot_score_and_time(x, y_score, y_time,subplot_pos, 
                               param_name= param_name, x_log_scale=log_flag)
        subplot_pos[2]+= 2
        
def launch_CV(X, y, y_name ,dict_models,dict_param_grid, cv = 5):
    '''

    Parameters:
    -----------------------------------------

    
    Returns:
    -----------------------------------------
    '''
    ## some warnings mostly on dual gap
    warnings.filterwarnings('ignore')
    ## init 
    dict_cv_results_= {}
    dict_best_params = {}
    for model_name in dict_models.keys(): 
        print(model_name)
        regressor = dict_models[model_name]
        param_grid = dict_param_grid[model_name]
        CV_regressor = GridSearchCV(regressor, param_grid, refit=True, cv = cv)
        CV_regressor.fit(X, y)
        dict_best_params[model_name] = CV_regressor.best_params_
        res = CV_regressor.cv_results_
        dict_cv_results_[model_name] = res
        
    with open(res_path+y_name+"_dict_CV_res_reg.pkl", 'wb') as fp:
        pickle.dump(dict_cv_results_, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(res_path+y_name+"_dict_CV_best_params.pkl", 'wb') as fp:
        pickle.dump(dict_best_params, fp, protocol=pickle.HIGHEST_PROTOCOL) 

