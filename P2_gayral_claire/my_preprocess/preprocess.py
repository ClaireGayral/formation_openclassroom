
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_path = "/home/clairegayral/Documents/openclassroom/data/"
filename = data_path+"en.openfoodfacts.org.products.csv"

def read_df(filename):
    ##
    ## GET INDEX OF NUTRI SCORE IN COLUMNS : 
    ##

    tmp = pd.read_csv(filename, sep = "\t", nrows = 1).columns
    col_index = np.where(tmp=="nutrition-score-fr_100g")[0][0]
    col_index

    ##
    ## OPEN FILE WITH ITERATOR CHUNKS SELECTING ROWS WITH INFO ON col_index : :
    ##

    def valid(chunks):
        for chunk in chunks:
            mask = ~chunk.iloc[:,col_index].isna().values
            yield chunk.loc[mask]            
    chunksize = 10 ** 4
    chunks = pd.read_csv(filename, sep = "\t", low_memory=False, 
                         chunksize=chunksize, header=None)
    df_original = pd.concat(valid(chunks))
    df = df_original.drop(0, axis=0)
    df.columns = df_original.loc[0,:]

    print("Number of variables : ", df.shape[1])
    print("Number of product selected : ", df.shape[0])
    return(df)

##
## drop columns with too many missing values 
##

def preprocess_drop_col_nan(df, nan_threshold): 
    # drop columns with more than "nan_threshold" missing values
    df = df.drop(df.columns[nan_repartition>nan_threshold], axis = 1)

def preprocess_data(data):
    if "ingredients_from_palm_oil_n" in data.columns :
        data.at[data["ingredients_from_palm_oil_n"] > 0, "ingredients_from_palm_oil_n"] = 1
        if ("ingredients_from_palm_oil" in data.columns) :
            data['ingredients_from_palm_oil'].fillna(data['ingredients_from_palm_oil_n'], inplace=True)
            data.drop("ingredients_from_palm_oil_n", inplace = True, axis = 1)
        else :
            data.rename(columns={"ingredients_from_palm_oil_n":"ingredients_from_palm_oil"}, inplace=True)


df = read_df(filename)
nan_repartition = df.isna().sum(axis=0)
nan_threshold = nan_repartition.mean()

preprocess_drop_col_nan(df,nan_threshold)
preprocess_data(df)

##
## select variables : 
## 

from my_preprocess import data_field2
data_field2.list_of_characteristics

list_of_characteristics = data_field2.list_of_characteristics
list_of_tags = data_field2.list_of_tags
list_of_ingredients = data_field2.list_of_ingredients
list_of_misc = data_field2.list_of_misc
list_of_nutri_facts = data_field2.list_of_nutri_facts

interest_var = pd.Index(["code","product_name","creator","countries",
                         "additives_n","ingredients_from_palm_oil",
                         "ingredients_that_may_be_from_palm_oil_tags"])
interest_var = interest_var.append(df.columns.intersection(list_of_nutri_facts))

data = df[df.columns.intersection(interest_var)].copy()

##
### GESTION DES TYPES DANS DATA !!
##

## STR 
str_var = list_of_characteristics  
str_var += list_of_tags
str_var += list_of_ingredients
str_var += list_of_misc
str_var = data.columns.intersection(str_var).values
data[str_var] = data[str_var].astype("str")

## FLOATS (and INTs)
float_var = list_of_nutri_facts
float_var += ["additives_n", "ingredients_from_palm_oil_n","ingredients_from_palm_oil_n",
              "ingredients_that_may_be_from_palm_oil_tags"]
float_var = data.columns.intersection(float_var).values
data[float_var] = data[float_var].astype("float")

## CATEGORY
data["creator"] = data[["creator","nutrition-score-fr_100g"]].astype("category")
