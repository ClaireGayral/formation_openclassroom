import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing

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
        figname = fig_name + "compare_regression"
        plt.savefig(res_path+"figures/"+figname+".jpg")
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
        res[lr_name]=metrics.r2_score(y_true = y_test, y_pred=y_pred, )
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
                  title = "Variables :", 
                  loc = "upper right",bbox_to_anchor=(1.5, 1))
    else : 
        plt.legend(var_names, title = "Variables :", 
                   loc = "upper right", bbox_to_anchor=(1.5, 1))
    ax.set_xlabel("alpha")
    ax.set_ylabel("Coefficient Values")
    ax.set_title("Regularization paths")
    if fig_name is not None : 
        figname = fig_name + "regul_paths_" + model_name + ".jpg" 
        plt.savefig(res_path+"figures/"+figname+".jpg")
        
        

def compute_LR_CV(X,y, dict_lr_model, alpha_values = np.logspace(-2, 2, 20), 
                  score_name= "r2", figsize = (8,5), fig_name = None) : 
    ## DROP MISSING VALUES IN y : 
    drop_index = y[y.isna()].index
    y_ = y.drop(drop_index, axis = 0)
    X_ = X.drop(drop_index, axis = 0)
    
    ## SPLIT DATA 
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=0.8)

    ## STANDARDIZE : 
    from sklearn.preprocessing import StandardScaler
    my_std = preprocessing.StandardScaler()
    my_std.fit(X_train)
    X_train_std = my_std.transform(X_train)
    X_test_std = my_std.transform(X_test)

    ## PLOTS :
    ## plot set of regulation parameter 
    plt.figure(figsize = figsize)
    res = compare_regressions(X_train_std, y_train,
                              dict_lr_model, alpha_values, 
                              score_name, fig_name)
    plt.show()
    ## plot regulation paths
    for model_name in dict_lr_model.keys():
        print(model_name," : ")
        plt.figure(figsize = figsize)
        plot_regul_paths(alpha_values, lm_model = dict_lr_model[model_name], 
                     X_ = X_train_std, y_ = y_train,
                     var_names = X.columns, best_alpha = res.loc[model_name,"best_alpha"],
                     fig_name = fig_name)
        plt.show()
    ## print results :
    return(res)


##
## ANOVA criteria
##

def eta_squared(x,y):

    ## y = num array
    ## x = categorical array
    
    ## drop nan individuals : 
    nan_index = np.concatenate((x[x.isna()].index.values,
                                y[y.isna()].index.values))
    x = x.drop(nan_index)
    y = y.drop(nan_index)
    ## init 
    moyenne_y = y.mean()
    classes = []
    ## loop on categories
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
        
##
## ANOVA PLOT 
##

my_color_set = list({ "forrest green":"#154406", "green":"#15b01a","sun yellow":"#ffdf22",
          "orange":"#f97306","lipstick red":"#c0022f","blue":"#0343df","shocking pink":"#fe02a2",
          "rust brown":"#8b3103","purple":"#7e1e9c","dark aquamarine":"#017371",
          "indigo":"#380282","grey blue" :"#6b8ba4","sky blue ":"#75bbfd",
          "pink":"#ff81c0","lavender":"#c79fef","neon red":"#ff073a",
          "goldenrod":"#fdaa48", "light salmon":"#fea993","salmon pink":"#fe7b7c",
          "magenta":"#c20078","teal":"#029386","olive green": "#677a04",
          "orangish brown":"#b25f03","almost black":"#070d0d", "silver" : "#c5c9c7",  #gris et noir
         }.values())

def sort_by_modality_mean(data, cat_var, num_var, sort):
    
    groups = []
    group_mean = []
    modalities = data[cat_var].values.categories
    for m in modalities:
        tmp = data[data[cat_var]==m][num_var]
        groups.append(tmp)
        group_mean.append(tmp.mean())
    ## sort by group values :
    if sort == True : 
        sort_index = np.argsort(group_mean)[::-1]
        groups = np.array(groups, dtype="object")[sort_index]
        modalities = modalities[sort_index]
    return(groups, modalities, sort_index)

def plot_boxplot(data,cat_var, num_var, sort = False, fig_name = None, dict_color_mod={}):
    
    groups, modalities, sort_index = sort_by_modality_mean(data,cat_var,num_var,sort)
    
    # Propriétés graphiques 
    medianprops = {'color':"black"}
    meanprops = {'marker':'o', 'markeredgecolor':'black','markerfacecolor':'blue'}
    # box plot : 
    boxplot = plt.boxplot(groups, labels=modalities, showfliers=False, medianprops=medianprops, 
                vert=True, patch_artist=True, showmeans=True, meanprops=meanprops)
    plt.title(str("Box plot "+num_var+" \nin " + cat_var +" modalities"))
    # add color : 
    for i,patch in enumerate(boxplot['boxes']) :
        mod = modalities[i]
        if (dict_color_mod.get(mod)==None): 
            color = "#6b8ba4"
        else : 
            color = dict_color_mod[mod]
        patch.set_facecolor(color)
        
    if np.any([type(mod) == str for mod in modalities]) :
        plt.xticks(rotation='vertical')    
    if fig_name is not None :
        plt.savefig(res_path+"figures/"+fig_name)

def get_different_clustering_color(original,cluster):
    dict_corresp_cluster_var = get_corresp_cluster_var(original, cluster)
    dict_color_mod = {}
    for i,list_var_in_cluster in enumerate(dict_corresp_cluster_var.values()):
        color = my_color_set[i]
        for var in list_var_in_cluster :
            dict_color_mod[var] = color
    return(dict_color_mod) 




##
## Cluster categorical vars w.r.t a numerical subspace
##

from copkmeans.cop_kmeans import cop_kmeans

########### inspired from https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
from scipy.spatial import distance

def compute_clustering_criterion(clusters,centers,Y):
    """
    Computes the AIC and BIC metric for a given clusters
    
    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn
    Y     :  multidimension np array of data points
    
    Returns:
    -----------------------------------------
    AIC,BIC value
    """
    #number of clusters
    K = max(clusters)+1
    # size of the clusters
    d = np.bincount(clusters)
    #size of data set
    n, p = Y.shape
    #compute variance for all clusters beforehand
#     cl_var = (1.0 / (N - K) / d) * 
    #sum of squared distances of samples to their closest cluster center
    inertia = float(sum([sum(distance.cdist(Y.values[np.where(clusters == i)], [centers[i]], 
             'euclidean')**2) for i in range(K)]))
#     const_term = 0.5 * m * np.log(N) * (d+1)
    AIC = float(inertia + 2*k*p)
    BIC = float(inertia + 2*np.log(n)*k*p)
    return(AIC,BIC)


def get_must_link(x):
    """
    Loop on categories to extract the pair of element that 
        should be in the same cluster
    
    Parameters:
    -----------------------------------------
    x: pd.Series(dtypes="category")
    
    Returns:
    -----------------------------------------
    list of 2-tuple 
    """
    must_link = []
    for category in x.cat.categories :
        group_index = x[x==category].index
        for element in group_index : 
            group_index = group_index.drop(element)
            for element2 in group_index :
                must_link.append((element,element2))
    return(must_link)

def cluster_categories(x,Y, k=5, must_link=None):
    """
    Computes the COP-Kmeans 
    
    Parameters:
    -----------------------------------------
    x : pd.Series(dtypes="category")
    Y : pd.Dataframe of data points (numerical)
    k : number of cluster 
    must_link : list of 2-tuple, if None, call get_must_link(x)
    
    Returns:
    -----------------------------------------
    clusters, centers in np.arrays
    """
    ## construction of input matrices : 
    input_matrix = Y.values
    ## list of tuple corresponding to the links we must keep : 
    if must_link is None :
        must_link = get_must_link(x) 
    ## launch cop kmeans : 
    clusters, centers = cop_kmeans(dataset=input_matrix, k=k, ml=must_link)
    clusters = np.array(clusters)
    centers = np.array(centers)
    return(clusters, centers)

#     res = data[num_vars].copy()
#     res.at[:,cat_var] = clusters
#     res[cat_var] = res[cat_var].astype("category")
#     return(res)

def loop_on_cat_var_cluster_category(data, num_vars, list_of_cat_var) : 
    """
    Computes the COP-Kmeans for different k on the categorical variables
    
    Parameters:
    -----------------------------------------
    data : pd.Dataframe 
    num_vars : list of numerical vars names (str) to use in Y
    list_of_cat_var : list of categorical vars name (str) to loop on
    
    Save in a file :
    -----------------------------------------
    AIC and BIC for multiple k
    """
    Y = data_log_transfo[num_vars]
    
    for cat_var in list_of_cat_var :
        x = data[cat_var]
        must_link = get_must_link(data[cat_var])
        min_k = 2
        K = len(data[cat_var].cat.categories)
        if (K>10) :
            max_k = int(np.floor(K)/2)
        
        else : 
            max_k = K
        cluster_criterion = pd.DataFrame(columns=["AIC","BIC"], 
                                         index = ["k_"+str(k) for k in range(min_k, max_k)],
                                         dtype=float)
        for k in range(min_k, max_k) :
            clusters, centers = cluster_categories(x,Y, k=k, must_link = must_link)
            cluster_criterion.at["k_"+str(k),:] = compute_bic(clusters,centers, Y)
        cluster_criterion.to_csv(res_path+"set_k_cluster_category_"+ cat_var +".csv")
        
def compare_boxplots_clustered_category(res_cluster, data,  cat_var, num_var = "CO2_emissions", 
                                    figsize = (20,15)) :
    """
    plot 2 boxplots to compare before and after clustering categories
    
    Parameters:
    -----------------------------------------
    res_cluster, data : pd.Dataframe 
    cat_var, num_var : str, resp. categorical and numerical vars names
    
    Returns:
    -----------------------------------------
    2 plt.subplot 
    """
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plot_boxplot(data, cat_var, num_var,sort=True)
    eta2 = eta_squared(x=data[cat_var], y=data[num_var])
    print(cat_var, "original : eta² =",np.round(eta2,2))

    plt.subplot(1,2,2)
    plot_boxplot(res_cluster,cat_var, num_var ,sort=True)
    eta2 = eta_squared(x=res_cluster[cat_var], y=res_cluster[num_var])
    print(cat_var, "clustered : eta² =",np.round(eta2,2))
    
    
def get_different_clustering_color(original,cluster):
    """
    extract a dictionnary of colors associated to each cluster, 
    color variables in the same cluster
    
    Parameters:
    -----------------------------------------
    original, cluster : pd.Dataframe 
    
    Returns:
    -----------------------------------------
    dictionnary of variables(keys) and colors (values)
    """
    dict_corresp_cluster_var = get_corresp_cluster_var(original, cluster)
    dict_color_mod = {}
    for i, clust_k in enumerate(dict_corresp_cluster_var.keys()):
        list_var_in_cluster = dict_corresp_cluster_var[clust_k]
        color = my_color_set[i]
        for var in list_var_in_cluster :
            dict_color_mod[var] = color
        dict_color_mod[clust_k] = color
    return(dict_color_mod) 

def compare_boxplots_clustered_category_COPkmeans(res_cluster, data,  
                                                  cat_var, num_var = "CO2_emissions", 
                                                  figsize = (20,15)) :
    """
    plot 2 boxplots to compare before and after clustering categories with cop-kmeans
    
    Parameters:
    -----------------------------------------
    res_cluster, data : pd.Dataframe 
    cat_var, num_var : str, resp. categorical and numerical vars names
    
    Returns:
    -----------------------------------------
    2 plt.subplot 
    """
    original = data[cat_var]
    cluster = res_cluster[cat_var]
    ## get colors : 
    dict_color_mod = get_different_clustering_color(original,cluster)
    
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plot_boxplot(data, cat_var, num_var,sort=True, dict_color_mod = dict_color_mod)
    eta2 = eta_squared(x=data[cat_var], y=data[num_var])
    print(cat_var, "original : eta² =",np.round(eta2,2))

    plt.subplot(1,2,2)
    plot_boxplot(res_cluster,cat_var, num_var, sort=True, dict_color_mod = dict_color_mod)
    eta2 = eta_squared(x=res_cluster[cat_var], y=res_cluster[num_var])
    print(cat_var, "clustered : eta² =",np.round(eta2,2))
    
    
def get_corresp_cluster_var(original,cluster):
    """
    From a COP-Kmeans clustering, get the original variable names in each cluster
    
    Parameters:
    -----------------------------------------
    original : 
    cluster : 
    
    Returns : 
    -----------------------------------------
    dictionnary s.t. keys = cluster names, values = list of original categorical variables
    """
    ## drop missing values ?
    drop_index = original[original.isna()].index
    original = original.drop(drop_index, axis = 0)
    cluster = cluster.drop(drop_index, axis = 0)
    
    dict_corresp_cluster_var={}
    for clust_k in cluster.cat.categories: 
        clust_index = cluster[cluster==clust_k].index
        dict_corresp_cluster_var[clust_k] = np.unique(original[clust_index].values)
        
    return(dict_corresp_cluster_var)

