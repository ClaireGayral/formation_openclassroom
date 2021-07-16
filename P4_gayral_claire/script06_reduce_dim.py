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

def draw_cluster_legend(ax2,clustering, corresp_color_dict):
    ## plot the legend with colored arrow
    # number of clusters : 
    K = len(clustering.values.categories)
    my_color = clustering.values.categories.map(corresp_color_dict)
    # plot parallel arrows :
    ax2.quiver(np.zeros(K),np.arange(0,K),np.ones(K),np.zeros(K),
               color = my_color)
    # plot legend text next to the respective arrow :
    for k in clustering.values.categories :
        cluster_var = get_str_vars(list(clustering[clustering == k].index.values))
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

def display_circles(pcs, n_comp, my_meth, axis_ranks, labels=None, 
                    label_rotation=0, lims=None, clustering = None, 
                    figsize = (9,6), fig_name=None):
    ## set coloration palette : 
    if clustering is not None : 
        my_color_set = ['#154406', '#15b01a', '#f97306', '#c0022f',
                        '#0343df', '#fe02a2', '#8b3103', '#7e1e9c', '#017371',
                        '#380282', '#6b8ba4', '#75bbfd', '#ff81c0', '#c79fef',
                        '#ff073a', '#fdaa48', '#fea993', '#fe7b7c', '#c20078',
                        '#029386', '#677a04', '#b25f03', '#070d0d', '#ffdf22']
        corresp_color_dict = dict(zip(clustering.values.categories, my_color_set))
        my_color = clustering.values.map(corresp_color_dict)

    else : 
        my_color = "grey"
    ## set global plot arguments : 
    plot_kwargs = {"alpha":1, "color":my_color}
    ## draw the correlation circle for my_meth : 
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:
            ## initialise figure
            figsize_x = list(figsize)[0]
            figsize_y = list(figsize)[1]
            # figsize = (figsize_x, figsize_y)
            if clustering is not None : 
                figsize_x = 2*figsize_x
                fig = plt.figure(figsize = (figsize_x, figsize_y))
                ## affichage de la legende du clustering en couleur : 
                ax2 = fig.add_subplot(1,2,2)
                draw_cluster_legend(ax2, clustering, corresp_color_dict)
                # initialisation de la figure "cercle"
                ax1 = fig.add_subplot(1,2,1)
            else : 
                fig = plt.figure(figsize = (figsize_x, figsize_y))
                ax1 = fig.add_subplot(1,1,1)

            ## détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            ## affichage des fleches :
            if pcs.shape[1] < 30 :
                ax1.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]), ## depart points
                           pcs[d1,:], pcs[d2,:], ## movement in each direction
                           angles='xy', scale_units='xy',**plot_kwargs)
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else: # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax1.add_collection(LineCollection(lines, axes=ax, alpha=.1, color=my_color))

            ## affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        ax1.text(x, y, labels[i], fontsize='14', ha='center', 
                                 va='center', rotation=label_rotation, color="blue", alpha=0.5)

            ## affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            ## définition des limites du graphique
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)

            ## affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')
            
            if hasattr(my_meth,"explained_variance_ratio_") : 
                ## nom des axes, avec le pourcentage d'inertie expliqué
                ax1.set_xlabel('F{} ({}%)'.format(d1+1, 
                                    round(100*my_meth.explained_variance_ratio_[d1],1)))
                ax1.set_ylabel('F{} ({}%)'.format(d2+1, 
                                    round(100*my_meth.explained_variance_ratio_[d2],1)))
            else : 
                ## nom des axes
                ax1.set_xlabel('F{}'.format(d1+1))
                ax1.set_ylabel('F{}'.format(d2+1))
            ax1.set_title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            if fig_name is not None :
                plt.savefig(res_path+"figures/"+fig_name+str(d1+1)+str(d2+1)+".jpg")

        # plt.show(block=False)
    
        
def display_factorial_planes(X_projected, n_comp, my_meth, axis_ranks, ind_labels=None, alpha=1, clustering = None, figsize = (12,10)):
    # args are as defined just above 
    plot_kwargs = {"marker":"x", "alpha":alpha, 's':20}#, "label" : clustering.values.categories}
    # set dict of color if clustering : 
    if clustering is not None : 
        my_color_set = ['#154406', '#15b01a', '#fffd01', '#f97306', '#c0022f', '#0343df', '#fe02a2', '#8b3103', 
                    '#7e1e9c', '#017371', '#380282', '#6b8ba4', '#75bbfd', '#ff81c0', '#c79fef', '#ff073a', 
                    '#fdaa48', '#fea993', '#fe7b7c', '#c20078', '#029386', '#677a04', '#b25f03', '#070d0d']
        corresp_color_dict = dict(zip(clustering.values.categories, my_color_set))

    for d1,d2 in axis_ranks:
        if d2 < n_comp:
            ax1 = "Axis"+ str(d1+1)
            ax2 = "Axis"+ str(d2+1)
            # initialisation de la figure       
            fig = plt.figure(figsize=figsize)
            if clustering is not None :
                for k in clustering.values.categories:
                    cluster_index = clustering[clustering==k].index
    #                 print(X_projected.loc[cluster_index, ax1])
                    plt.scatter(X_projected.loc[cluster_index, ax1],X_projected.loc[cluster_index, ax2], 
                                color=corresp_color_dict[k], label = k, **plot_kwargs)
                    plt.legend()

            else : 
                plt.scatter(X_projected[ax1], X_projected[ax2], **plot_kwargs)
            # affichage des labels des points
            if ind_labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, ind_labels[i],
                              fontsize='14', ha='center',va='center') 

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected.values[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            if hasattr(my_meth,"explained_variance_ratio_") : 
                ## nom des axes, avec le pourcentage d'inertie expliqué
                plt.xlabel('F{} ({}%)'.format(d1+1, 
                                    round(100*my_meth.explained_variance_ratio_[d1],1)))
                plt.ylabel('F{} ({}%)'.format(d2+1, 
                                    round(100*my_meth.explained_variance_ratio_[d2],1)))
            else : 
                ## nom des axes
                plt.xlabel('F{}'.format(d1+1))
                plt.ylabel('F{}'.format(d2+1))
            
            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            # plt.show(block=False)
            

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
