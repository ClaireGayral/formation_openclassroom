import numpy as np
import pandas as pd

energy_var =  ['energy-kj_100g', 'energy-kcal_100g', 'energy_100g',
               'fat_100g','saturated-fat_100g']

var_rescale_100g = ['monounsaturated-fat_100g','polyunsaturated-fat_100g',
                     'omega-3-fat_100g','omega-6-fat_100g', 'omega-9-fat_100g',
                     'trans-fat_100g','cholesterol_100g','carbohydrates_100g',
                     'sugars_100g','starch_100g','polyols_100g','fiber_100g', 
                     'proteins_100g','casein_100g','serum-proteins_100g',
                     'nucleotides_100g','sodium_100g','alcohol_100g',
                     'vitamin-a_100g','vitamin-d_100g','vitamin-e_100g',
                     'vitamin-k_100g','vitamin-c_100g','vitamin-b1_100g',
                     'vitamin-b2_100g','vitamin-pp_100g','vitamin-b6_100g',
                     'vitamin-b9_100g','vitamin-b12_100g','biotin_100g',
                     'pantothenic-acid_100g', 'silica_100g','bicarbonate_100g',
                     'potassium_100g','chloride_100g','calcium_100g',
                     'phosphorus_100g', 'iron_100g','magnesium_100g',
                     'zinc_100g','copper_100g', 'manganese_100g',
                     'fluoride_100g', 'selenium_100g','chromium_100g',
                      'molybdenum_100g','iodine_100g', 'caffeine_100g',
                     'taurine_100g', 'ph_100g','fruits-vegetables-nuts_100g']
                        
possible_val_dict = { # var_rescale_100g
                     'monounsaturated-fat_100g':[0,100],
                     'polyunsaturated-fat_100g':[0,100],
                     'omega-3-fat_100g':[0,100], 'omega-6-fat_100g':[0,100],
                     'omega-9-fat_100g':[0,100], 'trans-fat_100g':[0,100], 
                     'cholesterol_100g':[0,100],'carbohydrates_100g':[0,100],
                     'sugars_100g':[0,100], 'starch_100g':[0,100], 
                     'polyols_100g':[0,100],'fiber_100g':[0,100], 
                     'proteins_100g':[0,100], 'casein_100g':[0,100],
                     'serum-proteins_100g':[0,100], 'nucleotides_100g':[0,100],
                     'sodium_100g':[0,100],'alcohol_100g':[0,100], 
                     'vitamin-a_100g':[0,1], 'vitamin-d_100g':[0,1],
                     'vitamin-e_100g':[0,1], 'vitamin-k_100g':[0,1],
                     'vitamin-c_100g':[0,1],'vitamin-b1_100g':[0,1],
                     'vitamin-b2_100g':[0,1], 'vitamin-pp_100g':[0,1],
                     'vitamin-b6_100g':[0,1], 'vitamin-b9_100g':[0,1],
                     'vitamin-b12_100g':[0,1],'biotin_100g':[0,100],
                     'pantothenic-acid_100g':[0,10], 'silica_100g':[0,100],
                     'bicarbonate_100g':[0,10], 'potassium_100g':[0,100],
                     'chloride_100g':[0,100],'calcium_100g':[0,100],
                     'phosphorus_100g':[0,100], 'iron_100g':[0,100],
                     'magnesium_100g':[0,100],'zinc_100g':[0,100],
                     'copper_100g':[0,100], 'manganese_100g':[0,100],
                     'fluoride_100g':[0,100], 'selenium_100g':[0,100], 
                     'chromium_100g':[0,100], 'molybdenum_100g':[0,100], 
                     'iodine_100g':[0,100], 'caffeine_100g':[0,100], 
                     'taurine_100g':[0,100], 'ph_100g':[0,100],
                     'fruits-vegetables-nuts_100g':[0,100],
                      # energy_var : 
                     'energy-kj_100g':[0,25000],
                     'energy-kcal_100g':[0,5000], 'energy_100g':[0,30000],
                     'fat_100g':[0,100], 'saturated-fat_100g':[0,100],
                      # other
                     'additives_n':[0,35], 'ingredients_from_palm_oil': [0,1],
                     'carbon-footprint_100g':[0,5000],
                     'nutrition-score-fr_100g':[-20,50],'nutrition-score-uk_100g':[-5,25]}


def extract_irreg_errors_val(colname,possible_values, data):
    outliers_val = []
    col_values = data[colname].drop_duplicates().values
    ## check possible values : 
    min_value, max_value = possible_values
    for val in col_values :
        if ~np.isnan(val) :
            if (val < min_value) or (val > max_value):
                outliers_val.append(val)
#         else : 
#             print(sum(data[colname].isna()),"missing values")
#     print(len(outliers_val), "item values out of the intervall", possible_values)
    return outliers_val

def help_to_set_outliers_vals(df, colname, possible_vals):
    data = df.copy()
    fig = plt.figure(figsize=(15, 5))

    ## Histogramme global 
    ax = fig.add_subplot(1,3,1)
    nb_bins = min(50, len(np.unique(data[colname].dropna().values)))
    ax.hist(data[colname], bins = nb_bins, color='steelblue', density=True, edgecolor='none')
    ax.set_title("before removing outliers " + colname)

    outliers = extract_irreg_errors_val(colname,possible_vals, data = data)
    print("outliers products :",np.array(data.loc[data[colname].isin(outliers), "product_name"]), "\n")
    # replace outliers by np.nan : 
    data.at[data[colname].isin(outliers)] = np.nan

    ## Histogramme : 
    ax = fig.add_subplot(1,3,2)
    nb_bins = min(50, len(np.unique(data[colname].dropna().values)))
    ax.hist(data[colname], bins = nb_bins, color='steelblue', density=True, edgecolor='none')
    ax.set_title("after removing outliers " + colname)

    # plot values : 
    ax = fig.add_subplot(1,3,3)
    ax.plot(np.sort(data[colname]))
    return( outliers )

def rescale_outliers100g_val(data,possible_val_dict = possible_val_dict, concerned_var = var_rescale_100g):
    # from the hyp that the variable has been entered in mg instead of g -> rescale 
    # count_rescaled = pd.Series(np.zeros(len(data.columns)),index = data.columns)
    for colname in data.columns.intersection(concerned_var):
        min_value, max_value = possible_val_dict[colname] 
        is_outlier = (data[colname] < min_value) | (data[colname]>max_value) | ~data[colname].isna()  
        data.at[is_outlier, colname] = data.loc[is_outlier,colname]/1000 
    #     count_rescaled[colname] = sum(is_outlier)
    return(data)

def drop_outliers(data, possible_val_dict = possible_val_dict):
    ## drop outliers 
    for colname in data.columns.intersection(possible_val_dict.keys()):
        min_value, max_value = possible_val_dict[colname] 
        is_outlier = (data[colname] < min_value) | (data[colname]>max_value) #| (~data[colname].isna())  
        data.at[is_outlier, colname] = np.nan
    ## drop product with more than less than the median of missing values
    nan_repartition_row = data.isna().sum(axis=1)
    threshold_row = nan_repartition_row.mean()
    dropped_products = nan_repartition_row[nan_repartition_row>threshold_row].index
    data = data.drop(dropped_products, axis = 0)

    ## drop variables with more thant the 3rd quantile of missing values
    subset_var = data.columns[data.isna().sum(axis=0)>0]
    nan_repartition_col = data[subset_var].isna().sum(axis=0)
    threshold_col = nan_repartition_col.quantile(0.75)
    dropped_variables = nan_repartition_col[nan_repartition_col>threshold_col].index
    data = data.drop(dropped_variables, axis = 1)
    return(data)

