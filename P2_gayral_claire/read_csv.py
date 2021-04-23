import pandas as pd
import numpy as np

##
## META DATA ## 
##

data_path = "/home/clairegayral/Documents/openclassroom/data/"
filename = data_path+"en.openfoodfacts.org.products.csv"


## 
## open csv : 
##

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

def save_csv(filename):
    df_original = read_df(filename)
    df_original.to_csv(data_path+"projet2/df_original.csv")



