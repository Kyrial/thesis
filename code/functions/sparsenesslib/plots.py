import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import itertools
from scipy import linalg
import numpy as np
import math as math

def getCovariances(gmm):
    #covar = np.array();
    if gmm.covariance_type == "full":
        covariances = gmm.covariances_#[:][:2, :2]
    elif gmm.covariance_type == "tied":
        covariances = gmm.covariances_[:2, :2]
    elif gmm.covariance_type == "diag":
        #covariances = np.diag(gmm.covariances_[:][:2])
        covariances = list(map(lambda a: np.diag( a),gmm.covariances_[:][:2]))
    elif gmm.covariance_type == "spherical":
        covariances = list(map(lambda a: np.eye(gmm.means_.shape[1]) * a,gmm.covariances_[:]))
        #cov = numpy.array(listcov)
        #covariances = np.fromiter(map(lambda a: np.eye(gmm.means_.shape[1]) * a,gmm.covariances_[:]),dtype=np.float)
        
       # covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[:]
    return covariances

def plot_MultiGaussian(X, gm, index, title, X_MDS = None):
    Y_ = gm.predict(X)
    means= gm.means_
    #covariances = gm.covariances_
    covariances = getCovariances(gm)

    color_iter = itertools.cycle(["lightgreen", "c", "crimson", "darkviolet", "orangered", "olive"])
    splot = plt.subplot(2, 1, 1 + index)
    plt.tight_layout(h_pad=2)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        if type(X_MDS) is np.ndarray:
            plt.scatter(X_MDS[Y_ == i, 0], X_MDS[Y_ == i, 1], 0.8, color=color)
        else:
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)
        """
        #plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)
        # plt.scatter(X[Y_ == i, 4], X[Y_ == i, 5], 0.8, color="pink")
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        #ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        if math.isnan(angle):
            angle = 0

        for nsig in [1]: #[0.33,0.66, 1]:
            ell = (mpl.patches.Ellipse(mean, nsig * v[0], nsig * v[1],
                             180.0 + angle, color=color))
           
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.4)
            splot.add_artist(ell)
        """

    plt.xlim(X.min(), X.max())    
    plt.ylim(X.min(), X.max())   
    #plt.xticks(())
    #plt.yticks(())
    #plt.xlabel("U.A.")
    #plt.ylabel("U.A.")
    plt.title(title)
    plt.show()
    print('test')
    #logLikelihood(X)
    #logLikelihood(X,X[0])





def plotBIC(tabBIC, best_gmm, cv_types = ["spherical", "tied", "diag", "full"], n_components_range =7):
    bic = np.array(tabBIC)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_gmm
    bars = []
    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        bars.append(
            plt.bar(
                xpos,
                bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                width=0.2,
                color=color,
            )
        )
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
    plt.title("BIC score per model")
    xpos = (
        np.mod(bic.argmin(), len(n_components_range))
        + 0.65
        + 0.2 * np.floor(bic.argmin() / len(n_components_range))
    )
    plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
    spl.set_xlabel("Number of components")
    spl.legend([b[0] for b in bars], cv_types)
    #plt.show()