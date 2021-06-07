import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

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
