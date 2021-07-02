import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## 
## Plot hist fast
##

# y = pd.Series 
## plot hist : 
def plot_hist_y(y,fig_name = None):
    nb_bins = min(20, len(np.unique(y.values)))
    plt.hist(y, bins = nb_bins, color='steelblue', edgecolor='none')
    plt.title("Histogram of "+y.name)
    plt.xlabel("score")
    if fig_name is not None : 
        plt.savefig(res_path+"figures/"+fig_name)
    plt.show()
    
## plot 2 graphs : one of hist, and one of the repartition function
def repartition_and_hist(data,var,figsize=(10,5)):
    # data = pd.DataFrame
    # var = colname in data
    tmp = data.loc[~data[var].isna(), var]
    plt.figure(figsize=figsize)
    ## sorted plot
    plt.subplot(1,2,1)
    plt.plot(range(len(tmp)),tmp.sort_values())
    plt.xticks(rotation='vertical')
    plt.title(var)
    ## hist 
    plt.subplot(1,2,2)
    nb_bins = len(np.unique(tmp.values))
    plt.hist(tmp, bins=nb_bins, color='steelblue', edgecolor='none')
    plt.xticks(rotation='vertical')
    plt.title("Histogram of "+tmp.name)
    plt.show()

## plot gathering of hist/barplots
def plot_multi_hist(data, list_of_var=None, x_rotate = False, figsize=(18, 20), fig_name=None) : 
    if list_of_var is None :
        list_of_var = data.columns
    nb_line_plot = int(np.floor(len(list_of_var)/4)+1)
    fig = plt.figure(figsize=figsize)
    fig_count = 1
    for var in data.columns.intersection(list_of_var) :#data.columns.intersection(list_of_nutri_facts):
        ax = fig.add_subplot(nb_line_plot,4, fig_count)
        if isinstance(data[var].dtype, pd.api.types.CategoricalDtype) :
        #data.dtypes[var] == "category" :
            y_counts = data[var].value_counts()
            ax.bar(height=y_counts.values, x = y_counts.index)
            if type(y_counts.index[0])==str :
                plt.xticks(rotation='vertical')
        else :
            nb_bins = min(30, max(10, len(np.unique(data[var].dropna().values))))
            ax.hist(data[var], bins = nb_bins, density=True, edgecolor='none',
                   orientation = "vertical")
        ax.set_title(var)
        fig_count += 1
        plt.tight_layout()
    if fig_name is not None :
        plt.savefig(res_path+"figures/"+fig_name)
    plt.show()
    
    
##
## plot hist, density and boxplot 
##
    
def plot_y_hist_and_boxplot(y,my_title,subplot_pos1,subplot_pos2) : 
    plt.tight_layout(pad=2.0)
    ## hist with inferred density
    plt.subplot(*subplot_pos1,)
    plt.title(my_title)
    sns.histplot(y, bins=int(1 + np.log2(len(y))), color='steelblue', kde =True)
    plt.xlabel("values")
    plt.tight_layout(pad=1.0)
    ## boxplot bellow
    plt.subplot(*subplot_pos2,)
    sns.boxplot(data = y, orient="h")
    plt.xlabel("")  

def row_based_idx(subplot_pos):
    num_rows, num_cols, idx = subplot_pos
    return np.arange(1, num_rows*num_cols + 1).reshape((num_rows, num_cols)).transpose().flatten()[idx-1]

def convert_subplots_plot_bellow(subplot_pos) : 
    subplot_pos1 = subplot_pos.copy()
    subplot_pos2 = subplot_pos.copy()
    subplot_pos2[2] = int(subplot_pos2[2]+1)
    if subplot_pos[1]>1:
        subplot_pos1[2] = row_based_idx(subplot_pos1)
        subplot_pos2[2] = row_based_idx(subplot_pos2)        
    return(subplot_pos1,subplot_pos2)

def get_y_from_data(colname, data,  dict_var_log_transfo):
    log_scale = dict_var_log_transfo[colname]
    y = data[colname].copy()
    my_title = str("Distribution of variable "+colname)
    if log_scale :
        y[y<0] = np.nan
        y = np.log(y+1)
        my_title += " \nlog-transformed"
    return(y, my_title)

def plot_multi_hist_and_boxplot(data, list_of_var = None, dict_var_log_transfo=None, 
                                nb_fig_in_line = 1, figsize = (7,7) ):
    if list_of_var is None : 
        list_of_var = data.columns
    if dict_var_log_transfo is None :
        dict_var_log_transfo = {var : False for var in list_of_var}

    nb_lines = int(2 * np.ceil(len(list_of_var)/nb_fig_in_line))
    plt.figure(figsize=figsize)
    for fig_count,colname in enumerate(list_of_var) : 
        y, my_title = get_y_from_data(colname, data,  dict_var_log_transfo)

        subplot_pos = [nb_lines, nb_fig_in_line, 2*fig_count+1]
        subplot_pos1,subplot_pos2 = convert_subplots_plot_bellow(subplot_pos)

        plot_y_hist_and_boxplot(y,my_title,subplot_pos1,subplot_pos2)
        
        
##
## Other description 
##

## lorenz curve : 
def plot_lorenz_curve(y):
    n = len(y)
    y_rescaled = y - min(y)
    lorenz = np.cumsum(np.sort(y_rescaled)) / y_rescaled.sum()
    lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0
#     fig, ax = plt.subplots()
#     ax.axis('scaled')
    xaxis = np.linspace(0-1/n,1+1/n,n+1) 
    plt.plot(xaxis,lorenz,drawstyle='steps-post')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot([0,0],[-0.05,1.05], color="grey") # y axis 
    plt.plot([-0.05,1.05],[0,0], color="grey") # x axis
    plt.plot([-0.05,1.05],[-0.05,1.05], "--") # identity line
    plt.xlabel("Cumulative share of buildings")
    plt.ylabel("Cumulative share of score")
    plt.title("Lorenz curve for "+y.name)

    ## print AUC 
    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n # Surface sous la courbe de Lorenz. Le premier segment (lorenz[0]) est à moitié en dessous de 0, on le coupe donc en 2, on fait de même pour le dernier segment lorenz[-1] qui est à moitié au dessus de 1.
    S = 0.5 - AUC # surface entre la première bissectrice et le courbe de Lorenz
    gini = 2*S
    print(y.name, ":")
    print("\t gini =", gini)
    print("\t AUC =", AUC)

