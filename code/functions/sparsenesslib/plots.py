import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import itertools
from scipy import linalg
import numpy as np
import math as math
import pandas

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
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange",  "darkviolet", "olive"])

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
        + 0.55
        + 0.2 * np.floor(bic.argmin() / len(n_components_range))
    )
    plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
    spl.set_xlabel("Number of components")
    spl.legend([b[0] for b in bars], cv_types)

    #plt.show()

def plotPC(listPC, listBDD, layersName, title = "nombre PC par BDD par couche"):
    pc = np.array(listPC)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange",  "darkviolet", "olive"])
    bars = []
    # Plot the pc
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(listBDD, color_iter)):
        xpos = np.array(range(len(layersName)))*1.2 + 0.2 * (i - len(listBDD)/2)
        bars.append(
            plt.bar(
                xpos,
                #pc[i * len(layersName) : (i + 1) * len(layersName)],
                pc[i],
                width=0.2,
                color=color,
            )
        )
    plt.xticks(np.array(range(len(layersName)))*1.2 , layersName, rotation=90)
    plt.ylim([pc.min() * 1.01 - 0.01 * pc.max(), pc.max()+pc.max()*0.1])
    plt.title(title)

    spl.legend([b[0] for b in bars], listBDD)
    plt.show()
"""
    def plotCorrelation(listSpearman, listPearson, listBDD, layersName, title = "nombre PC par BDD par couche"):
        Spearman = np.array(listSpearman)
        Pearson = np.array(listPearson)
    
        color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange",  "darkviolet", "olive"])
        bars = []
        # Plot the pc
        plt.figure(figsize=(8, 6))
        spl = plt.subplot(2, 1, 1)


        for k, (cv_type, color) in enumerate(zip(listBDD, color_iter)):
            for i in [Pearson,SpSearman]:

                xpos = np.array(range(len(layersName)))*1.2 + 0.2* (i+1) * (i - len(listBDD)/2)
                bars.append(
                    plt.bar(
                        xpos,
                        #pc[i * len(layersName) : (i + 1) * len(layersName)],
                        pc[i],
                        width=0.2,
                        color=color,
                    )
                )
        plt.xticks(np.array(range(len(layersName)))*1.2 , layersName, rotation=90)
        plt.ylim([pc.min() * 1.01 - 0.01 * pc.max(), pc.max()+pc.max()*0.1])
        plt.title(title)

        spl.legend([b[0] for b in bars], listBDD)
        plt.show()
"""


def plotHist(AllLLH, name= "histogramme", max = 2):
    bin = np.linspace(AllLLH.min(), AllLLH.max(),200)
    for i, llh in enumerate(AllLLH):
        if  i <max:
            spl = plt.subplot(2, max//2, 1+i)
            hist = plt.hist(llh, bins=bin)
            plt.grid()
    plt.suptitle(name)
    plt.show()

def plotHist_fromFiles(AllHist, bin, name = "histo"):
    for i, hist in enumerate(AllHist):
        
        
        if i<10:
            spl = plt.subplot(2, 5, 1+i)
            if i==2:
                plt.title(name)
            plt.bar(bin,hist,width=bin[1]-bin[0])         
    plt.xticks(list(map(lambda val: round(val,1),bin[::50])))
    plt.show()

def plot_correlation(AllLLH, name = "", nameXaxis= "", nameYaxis=""):
    if len(AllLLH)>1:
        
        df = pandas.DataFrame(AllLLH)
        df = df.T
        corr_df_P = df.corr(method='pearson')
        import seaborn as sns
        plt.subplot(2, 2, 1)
        #plt.figure(figsize=(8, 6))
        sns.heatmap(corr_df_P,annot=True)
        plt.title("Correlation Pearson")
        plt.grid()

        corr_df_S = df.corr(method='spearman')
        
        plt.subplot(2, 2, 2)
        sns.heatmap(corr_df_S,annot=True)
        plt.title("Correlation Spearman")
        plt.grid()
        

        plt.subplot(2, 2, 3)
        a = sns.scatterplot(x= AllLLH[0],y= AllLLH[1]);
        plt.title("nuage de point")
        plt.xlabel(nameXaxis)
        plt.ylabel(nameYaxis)
        plt.grid()
        plt.suptitle(name)
        plt.show()


       
def plot_Grid_KDE(grid,bandwidths):
    scores = [val for val in grid.cv_results_["mean_test_score"]]
    plt.semilogx(bandwidths, scores)
    plt.xlabel('bandwidth')
    plt.ylabel('accuracy')  
    plt.title('KDE Model Performance')
    plt.show()