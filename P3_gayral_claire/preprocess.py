
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn import preprocessing

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter

##
## META DATA ## 
##
data_path = "/home/clairegayral/Documents/openclassroom/data/P3/"
res_path = "/home/clairegayral/Documents/openclassroom/res/P3/"


#####################################

##
## Duplicates treatment
##


simplefilter("ignore", ClusterWarning)

def get_index_merge_duplicates(data, float_var, threshold_clustering = 1.15):
    ## return a dict of indexes to merge, with key = row_name + cluster
    
    ## std var by var :
    data_float = data[float_var].copy()
    data_float.at[:,:] = preprocessing.StandardScaler().fit_transform(data_float)

    res = {}

    rows = data["OSEBuildingID"].drop_duplicates()
    for row in rows : 
        x = data[data.OSEBuildingID == row]
        row_index =  x.index

        ## if there is more than one row with the same name 
        if len(row_index) > 1 : 
            row_values = data_float.loc[row_index,:]
            row_dist = pd.DataFrame(nan_euclidean_distances(row_values),
                                  columns=row_index, index = row_index)
            Z = linkage(row_dist, "weighted")
            row_clustering = pd.Series(fcluster(Z, t=threshold_clustering), index = row_index)
            for k in np.unique(row_clustering.values):
                index_merge = row_clustering[row_clustering==k].index
#                 if len(index_merge) > 1 :
#                     res[row+str(k)]= index_merge
                res[row+str(k)]= index_merge
    return(res)

def drop_and_merge_duplicates(data):
    data_clean = data.copy()
    for row_name in res:
        merge_index = res[OSEBuildingID]
        first_index = merge_index[0]
        if len(merge_index)>1:
            row_duplicate = data.loc[merge_index].mean()
            row_duplicate.at["OSEBuildingID"] = row_name[:-1] # original name
            data_clean = data_clean.drop(merge_index.values, axis=0) 
            data_clean.at[first_index] = row_duplicate
            data_clean.loc[first_index]
    return(data_clean)


###################################

##
##
## select variables : 
## 
##

def select_columns(df, list_of_var):
    return(df[df.columns.intersection(list_of_var)])


##
## Plot heatmap of distance 
##

import seaborn as sns

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
