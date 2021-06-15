import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import model_selection
import time


## 
## Correlation matrix
##

def plot_heatmap_dist(row_dist):
    # index_sort = row_dist[row_dist.sum()==row_dist.sum().min()].index[0]
    index_sort = row_dist.sum().sort_values().index.values
    tmp = row_dist.loc[index_sort,index_sort]
    ax = sns.heatmap(
    #     (row_dist>1).sort_values(by = index_sort).sort_values(by = index_sort, axis = 1),
        tmp,
        vmin=0, vmax=1, center=0.5,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right')
    plt.show()


##
## Linear Regression  
##

def plot_score(alpha_values, score, label = None, best_alpha =None, score_name ="r2") :
    ax = plt.gca()
    ax.set_xscale("log")
    if sum(score<=-100)>1:
        subset = np.where(score>=-100)
        alpha_values= alpha_values[subset]
        score = score[subset]
        
    ax.plot(alpha_values, score, label = label)
    ax.set_xlabel("log(alpha)")
    ax.set_ylabel("mean of scores ("+ str(score_name)+")")
    ax.set_title("Choice of alpha with " +str(score_name))
    ax.set_xlim([0.9*min(alpha_values),1.1*max(alpha_values)])
    if best_alpha is not None : 
        ax.plot([best_alpha,best_alpha], [-100, 100], 
                color="grey", linestyle="dashed")

def launch_cv(model_name,lr_model, alpha_values,X_,y_, score_name="r2"):
    time_ref = time.time()
    ## compute Cross Validation :
    CV = model_selection.GridSearchCV(lr_model, param_grid={"alpha":alpha_values},
                                             scoring=score_name,cv=5)
    CV.fit(X_, y_)
#     print("temps d'exécution ",model_name," = ",(time.time() - time_ref)/5)
    exec_time = (time.time() - time_ref)/5
    return(CV,exec_time)

def compare_regressions(X_, y_, dict_lr_model, alpha_values, score_name="r2", fig_name=None):    
    ## init save res :
    min_score = {}
    max_score = {}
    res = pd.DataFrame(columns=["score", "execution_time", "best_alpha"])
    res.at["lr", :] = np.nan

    ## SIMPLE LINEAR REGRESSION  :
    time_ref = time.time()
    lr = model_selection.GridSearchCV(linear_model.LinearRegression(), param_grid={},
                                      scoring=score_name,cv=5)
    lr.fit(X_, y_)
    res.at["lr", "execution_time"] = (time.time() - time_ref)/5
    score_lr = lr.cv_results_["mean_test_score"].mean()
    res.at["lr", "score" ] = score_lr
    res.at["lr","best_alpha"]=None
    
    # LOOP ON REG ON lr_model_list 
    for model_name,lr_model in dict_lr_model.items():
        CV,execution_time = launch_cv(model_name,lr_model,alpha_values,X_,y_)
        ## extract CV results in dictionnaries :
        res.at[model_name,"score"] = CV.cv_results_['mean_test_score'].mean()
        best_alpha = CV.best_params_["alpha"]
        res.at[model_name,"best_alpha"] = best_alpha
        res.at[model_name,"execution_time"] = execution_time
        ## plot references : 
        min_score[model_name] = min(CV.cv_results_['mean_test_score'])
        max_score[model_name] = max(CV.cv_results_['mean_test_score'])
        ## plot scoring : 
        plot_score(alpha_values, CV.cv_results_['mean_test_score'], model_name, best_alpha, score_name)
        ## save CV result
    ## add linear regression R2 line : 
    plt.plot([alpha_values[0],alpha_values[-1]], [score_lr, score_lr], label = "linear regression")
    # plt.ylim([-0.1,1])
    plt.ylim([1.1*min(min_score.values())-0.05, 1.1*max(max_score.values())+0.05])
    plt.legend()
    if fig_name is not None : 
        plt.savefig(res_path+"figures/"+fig_name+".jpg")
    return(res)

def get_lm_score(X_,y_, X_test_std,y_test,dict_best_alpha):
    res = {}
    lr_model = linear_model.LinearRegression()
    lr_model.fit(X_, y_)
    res["LinearRegression"]=lr_model.score(X_,y_)
    
    for lr_name, alpha in dict_best_alpha.items():
        lr_model = dict_lr_model[lr_name]
        lr_model.set_params(alpha = alpha)
        lr_model.fit(X_,y_)
        y_pred = lr_model.predict(X_test_std)
        res[lr_name]=metrics.r2_score(y_true = y_test, y_pred=y_pred)
    return(res)


## REGULARIZATION PATH :
def plot_regul_paths(alpha_values, lm_model, X_, y_, 
                     var_names = None, best_alpha = None, 
#                      figsize=(7,7),
                     fig_name = None):
    regulation_paths = []
    for alpha in alpha_values:
        lm_model.set_params(alpha = alpha)
        lm_model.fit(X_,y_)
        coeffs = lm_model.coef_
        regulation_paths.append(coeffs)
    ## PLOT REG. PÄTHS :
    ax = plt.gca()
    ax.set_xscale("log")
    ax.plot(alpha_values, regulation_paths)
    ## VERTICAL LINE WITH THE BEST ALPHA :
    if best_alpha is not None : 
        ax.vlines(best_alpha, ymin = np.min(regulation_paths), ymax = np.max(regulation_paths), 
                  color = "black", linestyle="dashed", label="best_alpha")
    ## LEGEND :
    if var_names is not None :
        ax.legend(np.concatenate((np.array(var_names),['best alpha'])),
                   title = "Variables :", loc = "center right")
    else : 
        plt.legend(var_names, title = "Variables :")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Coefficient Values")
    ax.set_title("Regularization paths")
    if fig_name is not None : 
        plt.savefig(res_path+"figures/"+fig_name+".jpg")