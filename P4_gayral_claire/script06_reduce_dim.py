import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd 
import matplotlib._color_data as mcd
import random

##
## annexe 
## 

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

##
## PCA
## 

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
            ax1 = "axis"+ str(d1+1)
            ax2 = "axis"+ str(d2+1)
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
            
def plot_PCA_proj_of_clusters(X_proj, my_meth, axis_rank, ind_labels=None, 
                         alpha=1, clustering=None, figsize=(12,10)):
    ##                     
    ## first use in P4 
    ##                     
    plot_kwargs = {"marker":"x", "alpha":alpha, 's':10}#, "label" : clustering.values.categories}
    n_comp = max(list(sum(axis_ranks, ())))+1
    if clustering is None :
        clustering = pd.Series(np.ones(X_proj.shape[0]),
                               index = X_proj.index,dtype="category")
    ## add yellow in color to match with nutri-score : (yellow = '#ffdf22')
    my_color_set = ['#154406', '#15b01a', '#ffdf22', '#f97306', '#c0022f',
                    '#0343df', '#fe02a2', '#8b3103', '#7e1e9c', '#017371',
                    '#380282', '#6b8ba4', '#75bbfd', '#ff81c0', '#c79fef',
                    '#ff073a', '#fdaa48', '#fea993', '#fe7b7c', '#c20078',
                    '#029386', '#677a04', '#b25f03', '#070d0d', '#ffdf22']
    my_color_set = my_color_set * (X_proj.shape[0]//len(my_color_set) + 1) 
    corresp_color_dict = dict(zip(clustering.values.categories, my_color_set))
    plot_rank = [1,len(axis_ranks),1]

    for cluster in clustering.cat.categories:
        selected_index = clustering[clustering==cluster].index
        sub_X_proj = X_proj.loc[selected_index,:]
        count_fig = 1
        plt.figure(figsize=figsize)
        for d1,d2 in axis_ranks:
            if d2 < n_comp:
                plot_rank[2] = count_fig            
                plt.subplot(*plot_rank)
                ax1 = "axis"+ str(d1+1)
                ax2 = "axis"+ str(d2+1)
                # initialisation de la figure       

                plt.scatter(sub_X_proj.loc[:, ax1],sub_X_proj.loc[:, ax2], 
                            color=corresp_color_dict[cluster], label = cluster, **plot_kwargs)
                plt.legend()

                plt.xlim([-10,10])
                plt.ylim([-20,20])
                # affichage des lignes horizontales et verticales
                plt.plot([-100, 100], [0, 0], color='grey', ls='--')
                plt.plot([0, 0], [-100, 100], color='grey', ls='--')

                # nom des axes, avec le pourcentage d'inertie expliqué
                plt.xlabel('F{} ({}%)'.format(d1+1, round(100*my_pca.explained_variance_ratio_[d1],1)))
                plt.ylabel('F{} ({}%)'.format(d2+1, round(100*my_pca.explained_variance_ratio_[d2],1)))

                plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
                count_fig += 1 
        plt.show()
        


def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    #plt.show(block=False)
   
#
##
## NMF
##

def frobenius_func(y, y_pred):
    return(np.linalg.norm(y-y_pred,"fro"))


def pseudo_cv_reduce_dim(X_, my_meth, param_grid,my_score, cv = 5):
    ## MAP THE DICT OF LIST INTO LIST OF DICT :
    param_dirg = model_selection.ParameterGrid(param_grid)

    ## INITIALIZATION : 
    res = {} # dict of dict 
    res["params"]=[]
    for kwargs in param_dirg :
        res["params"].append(kwargs)
    dict_score = {}
    dict_time_fit = {}
    dict_time_predict = {}

    k_iter = 1
    ## SET FOLDS :
    kf = model_selection.KFold(n_splits = 5)
    CV_split_iterator = kf.split(X_) 

    ## LOOP ON FOLDS :
    for CV_train_range_index, CV_test_range_index in CV_split_iterator : 
        ## extract train
        train_index = X_.index[CV_train_range_index]
        train = X_.iloc[CV_train_range_index]
        ## LOOP ON PARAM NAMES (HERE ONLY 1)
        fold_key = "fold"+str(k_iter)
        ## init fold dict
        dict_score[fold_key] = []
        dict_time_fit[fold_key] = []
        dict_time_predict[fold_key] = []
        ## loop on different set of kwargs 
        for kwargs in param_dirg :
            ## SET PARAMS IN METH :
            my_meth.set_params(**kwargs)
            ## PREDICT TEST VALUES : 
            t = time.time()
            W = my_meth.fit_transform(train)
            dict_time_fit[fold_key].append(time.time() - t)
            t = time.time()
            H = my_meth.components_
            X_pred = np.dot(W,H)
            dict_score[fold_key].append(my_score(train, X_pred))
            dict_time_predict[fold_key].append(time.time() - t)
        k_iter += 1
    ## save in same shape as sklearn GridSearchCV     
    df_time_fit = pd.DataFrame(dict_time_fit)
    df_time_predict = pd.DataFrame(dict_time_predict)
    df_score = pd.DataFrame(dict_score)
    res["mean_fit_time"] = df_time_fit.mean(axis=1).values
    res["std_fit_time"] = df_time_fit.std(axis=1).values
    res["mean_score_time"] = df_time_predict.mean(axis=1).values
    res["std_score_time"] = df_time_predict.std(axis=1).values
    res["mean_test_score"] = df_score.mean(axis=1).values
    res["std_test_score"] = df_score.std(axis=1).values
    return(res)

def plot_coeffs(my_meth, X_, X_name= "X"):
    '''
    from dim reduceur, plot coefficients of the 2 first axis
    and return coefficients on whole axis
    
    Parameters:
    -----------------------------------------
    my_meth = sklearn.decomposition methode like PCA of NMF 
    X_ = pd.DataFrame() of data to be reduced
    
    Returns:
    -----------------------------------------
    pd.DataFrame of coefficients
    '''
    my_meth.set_params(**{"n_components": X_.shape[1]})
    my_meth.fit(X_)

    coeffs = pd.DataFrame(my_meth.components_, columns = X_.columns,
                          index = ["ax_"+str(k) for k in np.arange(1,my_meth.n_components+1)])
    for colname in coeffs.columns :
        plt.scatter(x = coeffs.loc["ax_1", colname], 
                    y = coeffs.loc["ax_2", colname],
                    label = colname)
    plt.xlabel("coeff axis 1")
    plt.ylabel("coeff axis 2")
    my_meth_name = str(my_meth).split("(")[0]
    plt.title(my_meth_name+" on "+ str(X_name),fontsize=14)

    plt.legend()
    return(coeffs)


##
## hierarchical clustering
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
