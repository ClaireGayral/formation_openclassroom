import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## 
## Univariate analysis for one numerical var
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

def plot_multi_hist(data, list_of_var, x_rotate = False, figsize=(15, 20), fig_name=None) : 
    nb_line_plot = int(np.floor(len(list_of_var)/4)+1)
    fig = plt.figure(figsize=(15, 20))

    fig_count = 1
    for var in list_of_var :#data.columns.intersection(list_of_nutri_facts):
        ax = fig.add_subplot(nb_line_plot,4, fig_count)
        nb_bins = min(20, len(np.unique(data[var].dropna().values)))
        ax.hist(data[var], bins = nb_bins, density=True, edgecolor='none')
        ax.set_title(var)
        fig_count += 1
        
    if fig_name is not None :
        plt.savefig(res_path+"figures/"+fig_name)
    plt.show()

## lorenz curve : 
def plot_lorenz_curve(y):
    n = len(y)
    y_rescaled = y - min(y)
    lorenz = np.cumsum(np.sort(y_rescaled)) / y_rescaled.sum()
    lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

    fig, ax = plt.subplots()

    ax.axis('scaled')
    xaxis = np.linspace(0-1/n,1+1/n,n+1) 
    plt.plot(xaxis,lorenz,drawstyle='steps-post')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot([0,0],[-0.05,1.05], color="grey") # y axis 
    plt.plot([-0.05,1.05],[0,0], color="grey") # x axis
    plt.plot([-0.05,1.05],[-0.05,1.05], "--") # identity line

    plt.xlabel("Cumulative share of buildings")
    plt.ylabel("Cumulative share of score")
    plt.title("Lorenz curve for the Energy Star score")

    plt.show()

    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n # Surface sous la courbe de Lorenz. Le premier segment (lorenz[0]) est à moitié en dessous de 0, on le coupe donc en 2, on fait de même pour le dernier segment lorenz[-1] qui est à moitié au dessus de 1.
    S = 0.5 - AUC # surface entre la première bissectrice et le courbe de Lorenz
    gini = 2*S
    print("gini =", gini)
    print("AUC =", AUC)

