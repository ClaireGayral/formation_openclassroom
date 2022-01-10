DATA_PATH = "/home/clairegayral/Documents/openclassroom/data/P8/forest/"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## used to select cat vars
#from sklearn.model_selection import train_test_split
#from sklearn.inspection import permutation_importance
#from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


##
## import data 
##

df_train = pd.read_csv(DATA_PATH+"/train.csv", index_col=0)
df_test = pd.read_csv(DATA_PATH+"/test.csv", index_col=0)
target_train = df_train.loc[:,"Cover_Type"]
df_train = df_train.loc[:,[col for col in df_train.columns if col != "Cover_Type"]]

soil_vars = ['Soil_Type' +str(k) for k in range(1,41)]
area_vars = ['Wilderness_Area' +str(k) for k in range(1,5)]
cat_vars = soil_vars + area_vars
num_vars = [col for col in df_train.columns if col not in cat_vars]

## global variables : 
cat_vars_selected = ['Wilderness_Area4', 'Soil_Type10', 'Soil_Type38', 'Soil_Type39',
           'Soil_Type40', 'Soil_Type4', 'Soil_Type30', 'Soil_Type23',
           'Wilderness_Area3', 'Soil_Type2', 'Wilderness_Area1', 'Soil_Type13',
           'Soil_Type22', 'Soil_Type17', 'Soil_Type3', 'Soil_Type11',
           'Soil_Type35', 'Soil_Type24']

num_var_selected = [['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology',
                     'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 
                     'Hillshade_3pm','Horizontal_Distance_To_Fire_Points']]
                   
##
## Preprocessing
##

## categorical data : 
def get_cat_X(X_):
    return(X_.loc[:,cat_vars_selected])
    
## numerical data :

def get_num_X(X_):
    X_num = X_.loc[:,num_vars_selected]
    return(X_num)
    
def preprocess1(X_, calcul_hloc_pca=False):
    ''' 
    Concatenate num and cat preprocess 
    
    Parameters:
    -----------------------------------------
    X_ : pd.DataFrame, from lecture of data 
            (train or test csv data)
    calcul_hloc_pca : bool, should the hloc pca be recomputed 
                      be caution : do it only for train set 
    
    Returns:
    -----------------------------------------
    '''   
    X_num = get_num_X(X_)
    X_cat = get_cat_X(X_)
    X_complete = pd.concat([X_cat, X_num], axis=1)
    return(X_complete)    

def preprocess2(X_train, X_test):
    ''' 
    From the 1st preprocess result, apply statistical preprocess
    
    Parameters:
    -----------------------------------------
    X_train : pd.DataFrame, df_train with preprocess 1
    X_test : pd.DataFrame, df_test with preprocess 1
   
    Returns:
    -----------------------------------------
    res_train,res_test : the 2 pd.DataFrame of preprocessed data
    '''
    
    ## fit std model on X_train :
    my_std = StandardScaler()
    my_std.fit(X_train)
    X_train_std = my_std.transform(X_train)
    X_test_std = my_std.transform(X_test)
    
    ## fit pca projection on X_train standardized:
    n_compo = X_train.shape[1]
    my_pca = PCA(n_components=n_compo)
    my_pca.fit(X_train_std)
    res_train = my_pca.transform(X_train_std)
    res_train = pd.DataFrame(res_train, index=X_train.index, 
                             columns=["ax_"+str(k+1) for k in range(n_compo)])
    res_test = my_pca.transform(X_test_std)
    res_test = pd.DataFrame(res_test, index=X_test.index, 
                             columns=["ax_"+str(k+1) for k in range(n_compo)])
    return(res_train,res_test)

##
## Support Vector Machine classification with rbf kernel:
##
def __main__(): 
    ## preprocess 
    X_train, X_test = preprocess2(preprocess1_train,preprocess1_test)
    clf = SVC(kernel="rbf", C=100)
    ## train 
    clf.fit(X_train, target_train)
    prediction = clf.predict(X_test)
    return(prediction)
