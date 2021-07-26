import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cluster_scatter(df, labels, figsize=(10,6)):
    ## df = pd.DataFrame
    ## labels = pd.Series with same index as df
    plt.rcParams['figure.figsize'] = figsize
    ax = plt.axes(projection='3d')
    X = df.recency
    Y = df.frequency
    Z = df.monetary_value
    plt.xlabel("Recency")
    plt.ylabel("Frequency")
    ax.set_zlabel("Monetary value")
    for label in np.unique(labels):
        customers = labels[labels==label].index
        x = X.loc[customers]
        y = Y.loc[customers]
        z = Z.loc[customers]
        ax.scatter3D(x,y,z,  label="cluster_"+str(label))#, cmap='Greens')
    ax.legend(loc="upper right",bbox_to_anchor=(1.25,0.75))
    return(ax)
    
def plot_projection_on_frequency_values(df, labels, frequencies):
    count_fig = len(frequencies)+1
    plt.figure(figsize=(5*count_fig,5))
    for k in range(len(frequencies)):
        plt.subplot(1,count_fig,k+1)
        freq = frequencies[k]
        X_freq = df[df["frequency"]==freq]
        for label in np.unique(labels):
            customers = labels[labels==label].index
            customers = X_freq.index.isin(customers)
            x = X_freq.recency.loc[customers]
            y = X_freq.monetary_value.loc[customers]
            plt.scatter(x,y,label=label)
            plt.title("Projection on Frequency = "+str(freq))
            plt.xlabel(x.name)
            plt.ylabel(y.name)
    plt.subplot(1,3,k+2)    
    X_freq = df[df["frequency"]>freq]
    for label in np.unique(labels):
        customers = labels[labels==label].index
        customers = X_freq.index.isin(customers)
        x = X_freq.recency.loc[customers]
        y = X_freq.monetary_value.loc[customers]
        plt.scatter(x,y,label=label)
    plt.title("Projection on Frequency > "+str(freq))
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.show()
    
    
    

### modified code from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html

import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture
def plot_set_params_gmm(X, n_components_range = range(4, 13)):
    lowest_bic = np.infty
    bic = []
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin()+min(n_components_range)-1.12,
                  len(n_components_range)) + .65 +.2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    nb_colors = clf.get_params()["n_components"]
    clf_colors = sns.color_palette("viridis",nb_colors)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, 
                                               clf.covariances_,
                                               clf_colors)):
        cov_type = clf.get_params()["covariance_type"]
        if cov_type in ["diag", "spherical"]:
            v = cov
            w = np.zeros([cov.shape[0],cov.shape[0]])
            for index in range(cov.shape[0]) : w[index,index]=1.0
        else : 
            v, w = linalg.eigh(cov)

        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi 
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(f'Selected GMM: {best_gmm.covariance_type} model, '
              f'{best_gmm.n_components} components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()
