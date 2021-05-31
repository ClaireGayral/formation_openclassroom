import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn import preprocessing

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter

##
## Duplicates treatment 
##

#################################################################
## using a hierarchical clustering to merge similar duplicates ##
#################################################################
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
