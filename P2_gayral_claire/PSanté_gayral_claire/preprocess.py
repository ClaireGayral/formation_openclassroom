
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

##
## META DATA ## 
##
data_path = "/home/clairegayral/Documents/openclassroom/data/"
filename = data_path+"en.openfoodfacts.org.products.csv"

##
## drop columns with too many missing values 
##

def preprocess_drop_col_nan(df, nan_threshold): 
    # drop columns with more than "nan_threshold" missing values
    nan_repartition = df.isna().sum(axis=0)
    df = df.drop(df.columns[nan_repartition>nan_threshold], axis = 1)
    return(df)
##
## select variables : 
## 

def select_columns(df, list_of_var):
    return(df[df.columns.intersection(list_of_var)])

##
### GESTION DES TYPES DANS DATA !!
##
def set_dtypes(data) :
    import list_from_data_field 
    ## STR 
    str_var = list_from_data_field.list_of_characteristics
    str_var += list_from_data_field.list_of_tags
    str_var += list_from_data_field.list_of_ingredients
    str_var += list_from_data_field.list_of_misc
    str_var = data.columns.intersection(str_var).values
    data[str_var] = data[str_var].astype("str")

    ## FLOATS (and INTs)
    float_var = list_from_data_field.list_of_nutri_facts
    float_var += ["additives_n", "ingredients_from_palm_oil","ingredients_from_palm_oil_n"]
    float_var = data.columns.intersection(float_var).values
    data[float_var] = data[float_var].astype("float")
    
    ## CATEGORY
    data["creator"] = data[["creator","nutrition-score-fr_100g"]].astype("category")
    return(data)

##
## Special column treatment : 
##
def merge_palm_oil_cols(data):
    if "ingredients_from_palm_oil_n" in data.columns :
        data.at[data["ingredients_from_palm_oil_n"] > 0, "ingredients_from_palm_oil_n"] = 1
        if ("ingredients_from_palm_oil" in data.columns) :
            data['ingredients_from_palm_oil'].fillna(data['ingredients_from_palm_oil_n'], inplace=True)
            data.drop("ingredients_from_palm_oil_n", inplace = True, axis = 1)
        else :
             data.rename(columns={"ingredients_from_palm_oil_n":"ingredients_from_palm_oil"}, inplace=True)
        data["ingredients_from_palm_oil"] = data["ingredients_from_palm_oil"].astype("float")
    return(data)
 
def extract_irreg_errors_val(colname,possible_values, data):
    outliers_val = []
    col_values = data[colname].drop_duplicates().values
    ## check possible values : 
    min_value, max_value = possible_values
    for val in col_values :
        if ~np.isnan(val) :
            if (val < min_value) or (val > max_value):
                outliers_val.append(val)
        else : 
            print(sum(data[colname].isna()),"missing values")
    print(len(outliers_val), "item values out of the intervall", possible_values)
    return outliers_val

##
## Duplicates treatment
##

from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn import preprocessing

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

def get_index_merge_duplicates(data, float_var, threshold_clustering = 1.15):
    ## return a dict of indexes to merge, with key = product_name + cluster
    
    ## std var by var :
    data_float = data[float_var].copy()
    data_float.at[:,:] = preprocessing.StandardScaler().fit_transform(data_float)

    res = {}

    products = data["product_name"].drop_duplicates()
    for prod_name in products : 
        x = data[data.product_name == prod_name]
        prod_index =  x.index

        ## if there is more than one product with the same name 
        if len(prod_index) > 1 : 
            prod_values = data_float.loc[prod_index,:]
            prod_dist = pd.DataFrame(nan_euclidean_distances(prod_values),
                                  columns=prod_index, index = prod_index)
            Z = linkage(prod_dist, "weighted")
            prod_clustering = pd.Series(fcluster(Z, t=threshold_clustering), index = prod_index)
            for k in np.unique(prod_clustering.values):
                index_merge = prod_clustering[prod_clustering==k].index
                res[prod_name+str(k)]= index_merge
    return(res)