import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd 
import matplotlib._color_data as mcd
import random

def get_str_vars(list_of_var):
    # from a list of str, return a sentence
    # if the line is too long (sup to 40), cut
    tmp = list_of_var.copy()
    res = ""
    len_line = 0
    while tmp : 
        var = tmp.pop()
        res = res+ var +str(", ")
        len_line += len(var)
        if len_line > 40:
            res = res +"\n"
            len_line = 0
    return(res)

def draw_cluster_legend(ax2,cluster,my_color_set):
    ## plot the legend with colored arrow
    # number of clusters : 
    K = max(cluster)+1
    # plot parallel arrows :
    ax2.quiver(np.zeros(K),np.arange(0,K),np.ones(K),np.zeros(K),
               color = my_color_set[:K])
    # plot legend text next to the respective arrow :
    for k in range(K):
        cluster_var = get_str_vars(list(cluster[cluster == k].index.values))
        ax2.text(0.2, k , str(cluster_var), fontsize='11',
                 ha='left', va='center' , alpha=1)
    # set limits : 
    ax2.set_xlim([-0.1,2])
    ax2.set_ylim([-1.1, K+0.1])
    ax2.set_title("Clustering legend")
    # remove axis :
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    plt.axis("off")
    return(ax2)
# fig2, ax2 = plt.subplots(1,1)
# draw_cluster_legend(ax2, cluster)
# plt.show()

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None, cluster = None):
    ## draw the correlation circle for pca : 
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:
            fig = plt.figure(figsize = (18,6))
            # initialisation de la figure
            ax1 = fig.add_subplot(1,2,1)

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # gestion de la couleur si cluster : 
            if cluster is not None : 
                my_color_set = ['#c0022f', '#ff073a', '#f97306', '#fdaa48', 
                        '#fffd01', '#fea993', '#fe7b7c', '#ff81c0', '#fe02a2', 
                        '#c20078', '#c79fef', '#7e1e9c', '#380282', '#0343df', 
                        '#75bbfd', '#6b8ba4', '#029386', '#017371', '#15b01a', 
                        '#677a04', '#b25f03', '#8b3103', '#070d0d', '#c5c9c7']
                my_color = [my_color_set[i] for i in cluster.values]
            else : 
                my_color = "grey"


            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                ax1.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                           pcs[d1,:], pcs[d2,:], angles='xy', scale_units='xy', 
                           scale=1, color=my_color)
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax1.add_collection(LineCollection(lines, axes=ax, alpha=.1, color=my_color))

            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        ax1.text(x, y, labels[i], fontsize='14', ha='center', 
                                 va='center', rotation=label_rotation, color="blue", alpha=0.5)

            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            ax1.set_xlabel('F{} ({}%)'.format(d1+1, 
                                round(100*pca.explained_variance_ratio_[d1],1)))
            ax1.set_ylabel('F{} ({}%)'.format(d2+1, 
                                round(100*pca.explained_variance_ratio_[d2],1)))
            ax1.set_title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))

            ## affichage de la legende du clustering en couleur : 
            if cluster is not None : 
                ax2 = fig.add_subplot(1,2,2)
                draw_cluster_legend(ax2, cluster,my_color_set)
            plt.show(block=False)
            
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
            

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
##
## TP hierarchical clustering
## 

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(Z, names, figsize = (10,25)):
    plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()
