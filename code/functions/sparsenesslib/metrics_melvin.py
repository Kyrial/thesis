#####################################################################################
# LIBRAIRIES:
#####################################################################################
#public librairies
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
import sparsenesslib.plots as plots
import itertools
import time
import statsmodels.api as sm
import scipy.optimize as opt
import os
from sklearn.manifold import MDS
from scipy.ndimage import gaussian_filter1d
from scipy import linalg
from scipy.spatial import distance
from scipy import stats

from sklearn import decomposition
from sklearn.decomposition import IncrementalPCA
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn import metrics

import csv


###############



def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]


def SilouhetteCoef(X, showGraphe=False, verbose = False):
    print("calculs Silouhette Coef");
    maxCluster = 20
    n_clusters=np.arange(2, maxCluster)
    sils=[]
    sils_err=[]
    iterations=10
    for n in n_clusters:
        if verbose:
            verbosePourcent(n, maxCluster)
        tmp_sil=[]
        for _ in range(iterations):
            gmm=GaussianMixture(n, n_init=2).fit(X) 
            labels=gmm.predict(X)
            sil=metrics.silhouette_score(X, labels, metric='euclidean')
            tmp_sil.append(sil)
        val=np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
        err=np.std(tmp_sil)
        sils.append(val)
        sils_err.append(err)
    if showGraphe:
        plt.errorbar(n_clusters, sils, yerr=sils_err)
        plt.title("Silhouette Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("Score")
        plt.show()


def MultiDimensionalScaling(X):  
    """! Calculs la MDS
    @param X array de dimension N
    @return retourne un tableau de dimension 2
    """
    mds = MDS(random_state=0,n_jobs= -1)
    X_MDS =mds.fit_transform(X)
    return X_MDS


def BIC(X, verbose = False, plot = True, nbMaxComp = 20):
    """! Bayesian Information Criterion
    """
    lowest_bic = np.infty
    bic = []

    n_components_range = range(1, nbMaxComp)
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            if verbose:
                verbosePourcent(n_components*(cv_types.index(cv_type)),len(n_components_range)*len(cv_types) )
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(
                n_components=n_components, covariance_type=cv_type, 
             #   random_state = 1
            )
            gmm.fit(X) 
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm  
    if plot:
        plots.plotBIC(bic, best_gmm, n_components_range = n_components_range)
    return best_gmm


#gm = BayesianGaussianMixture(n_components =10, n_init = 2, weight_concentration_prior_type ="dirichlet_process", weight_concentration_prior =0.0000000001).fit(X_scale)



def getMultigaussian(X, plot= [True,True], name ="Gaussian Mixture", index = 1, nbMaxComp = 50):
    """! a partir d'un array, calcul le BIC, si plot, appelle MultiDimensionalScaling(X) afin d'obtenir un Array en 2D pour l'afficher
    @param X array de dimension N
    @param [facultatif] plot Boolean, affiche ou nom le graphique
    @param [facultatif] name string, titre pour le Plots
    @param [facultatif] index integer, pour la position dans le Plots
    @param renvoie la mixureGaussian
    """

    # Centrage et Réduction
    std_scale = preprocessing.StandardScaler().fit(X)
    X = std_scale.transform(X)

    #gm = GaussianMixture(
     #           n_components=1 
             #   random_state = 1
      #      )
    #gm.fit(X) 
    gm = BIC(X, verbose = False, plot = plot[0], nbMaxComp = nbMaxComp)
    plt.show()

    if plot[1]:
        X_MDS = MultiDimensionalScaling(X) 
        plots.plot_MultiGaussian(llh, index, name, X_MDS)
    return gm




def writeLikelihood(LLH,pathLLH, layer):
    """! a partir d'un array de LLH, écris le fichier correspondant
    @return array de LLH
    """
    df = pandas.DataFrame(LLH)
    df = df.transpose()
        
    os.makedirs(pathLLH, exist_ok=True)
        #l'enregistrer dans results, en précisant la layer dans le nom
    df.to_csv(pathLLH+"/"+"LLH_"+layer)
    return LLH



def DoMultipleLLH(gmm_kde, X, nbe):
    AllLLH = []
    for i in range(nbe):
        gmm_kde.fit(X)
        LLH = gmm_kde.score_samples(X); #récupère LLH
        AllLLH.append(np.transpose(LLH))
        #LLH_tr = np.transpose([LLH]) #transpose
        #std_scale = preprocessing.StandardScaler().fit(LLH_tr) #centrer reduit
        #LLH_tr = std_scale.transform(LLH_tr)
        #AllLLH.append(np.transpose(LLH_tr)[0])

#    AllLLH = np.array(AllLLH)
#    plots.plot_correlation(AllLLH
    return np.array(AllLLH)

def chooseBestComposante(allLLH):
    median = np.array(np.median(allLLH, axis=0))
    indexTab=[]
    for repetition in allLLH:
        similarite = repetition==median
        indexTab.append(similarite.sum())

    indexTab = np.array(indexTab)
    print(indexTab.max())

    max = indexTab.argmax()
    result = np.array([allLLH[max]])
    return result


def doHist(AllLLH, plot = False, name = "histogramme"):
    if isinstance(AllLLH, list):
        AllLLH = np.array(AllLLH)
    bin = np.linspace(AllLLH.min(), AllLLH.max(),500)
    allHist = []
    legend = []
    for i, llh in enumerate(AllLLH):
        
        hist = np.histogram(llh, bins=bin)
        legend = np.transpose(hist)[1]
        allHist.append(np.transpose(hist)[0])
    
    if plot:
        plots.plotHist(AllLLH, name, max = 2)
    #if len(AllLLH)>1:
    #    compareValue(AllLLH[0], AllLLH[1])
    #    CompareOrdre(AllLLH[0], AllLLH[1])
    return allHist, legend


def writeHist(allHist, legend, path,name):
   
    df = pandas.DataFrame(allHist)
    #df = df.transpose()
    df.columns = legend[0:-1]
    #os.makedirs(pathData+"results"+"/"+bdd+"/"+"LLH", exist_ok=True)
        #l'enregistrer dans results, en précisant la layer dans le nom
    os.makedirs(path, exist_ok=True)
    df.to_csv(path+"/"+name)

def CompareOrdre(x, y, name = "compare"):
    ordre_X = np.argsort(x)
    ordre_Y = np.argsort(y)
    compareValue(ordre_X, ordre_Y, name )

def compareValue(X, Y, name = "compare"):
    
    val = abs(X - Y)
    bin = np.linspace(val.min(), val.max(),1000)
    hist = plt.hist(val, bins=bin)
    plt.grid()
    plt.title(name)
    plt.show()

def spearman(x,y):
    s = stats.spearmanr(x, y)
    print(s)
    return s
def pearson(x,y):
    s = stats.pearsonr(x, y)
    print(s)
    return np.array(s)

def doVarianceOfGMM(gmm, X, plot = False):
    allLLH = DoMultipleLLH(gmm, X, 2)

    allHist, legend = doHist(allLLH)

    writeCSV=False
    if writeCSV:
        df = pandas.DataFrame(allHist)
        #df = df.transpose()
        df.columns = legend[0:-1]
        #os.makedirs(pathData+"results"+"/"+bdd+"/"+"LLH", exist_ok=True)
            #l'enregistrer dans results, en précisant la layer dans le nom
        df.to_csv("./test.csv")

    plt.show()

    #for i in range(10):
        #plt.hist(All)

    #for i in range(len(AllLLH)):
        #LLH_for_i =  AllLLH[:,i]
        
    allVar = np.var(AllLLH, axis=0) # récup la variance intraImage
    varExtra = np.var(AllLLH, axis=1)

    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange",  "darkviolet", "olive"])

    if plot:
       
        plt.plot(range(AllLLH.shape[1]),allVar)
        plt.title(r'variance intra image')
        plt.grid()
        plt.show()

        plt.plot(range(AllLLH.shape[0]),varExtra)
        plt.title(r'variance extra image')
        plt.grid()
        plt.show()
        print("finish")
    allRatio = ratioArray(allVar,varExtra)
    for i, color in zip(range(allRatio.shape[1]), color_iter):
        plt.plot(range(len(varExtra)), allRatio[:,i],color)
    plt.title(r'ratio par image, pour chaque Gmm')
   # plt.xlabel(r'$\iterations de GMM$')
    #plt.ylabel(r'$\ratio varIntra par varExtr$')
    plt.grid()
    plt.show()
      #  splot = plt.subplot(2, 1, 2)
   # for i, color in zip(range(allRatio.shape[1]), color_iter):
    #    plt.plot(range(len(varExtra)), allRatio[:,i],color)
    
    
   # plt.show()


def ratio(a, b):
    a = float(a)
    b = float(b)
    if b == 0:
        return a
    return ratio(b, a % b)
#returns a string with ratio for height and width
def get_ratio(a, b):
    r = ratio(a, b)
    return float((a/r) / (b/r))
def ratioArray(x, y):
    val = np.array(
        list( map(lambda b :
               list(map(lambda a :
                        get_ratio(a,b),
                        x)) ,
               y )))
    return val


def distToCentroid(X, variance, name =r'distance to centroid' ):
    centroid = np.mean(X, axis =0)
    v = variance[0]
    #covar = np.cov(X, aweights  = variance[0])

    #val =distance.mahalanobis(centroid,X[0], linalg.inv(covar))
    #print(val)

    tabDist = []
    for img in X:
        dist = 0;
        for i, centr, var in zip(img, centroid, variance[0]):
            dist += ((i**2 + centr**2)**.5)*var
            
        tabDist.append(dist);
    tabDist = np.array(tabDist)
    #print(tabDist)

    plt.plot(range(len(tabDist)),tabDist)
    plt.title(name)
    plt.grid()
    plt.show()

    
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

def findBandWith(x, intervalle, profondeur =4):

    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': intervalle},
                        #cv=LeaveOneOut()
                        #cv=is 5-Fold validation (default)
                        )
    grid.fit(x);
    tailleBande = grid.best_params_
    print("taille bande = ", tailleBande['bandwidth'],"\n")
    plots.plot_Grid_KDE(grid,intervalle)
    if profondeur > 0:
        newIntervalle = (intervalle.max() - intervalle.min())/15 
        return findBandWith(x, np.linspace(max(10**-1,tailleBande['bandwidth']-newIntervalle), tailleBande['bandwidth']+newIntervalle, 80), profondeur-1)
    else:
        return tailleBande
###
def KDE(x, recursion = False):
    # Centrage et Réduction
    std_scale = preprocessing.StandardScaler().fit(x)
    x = std_scale.transform(x)

    
    bandwidths = np.linspace(10**-1, 100, 200)
    #bandwidths = np.linspace(10**-2, 10**-1, 200)
    #bandwidths = 10 ** np.linspace(-1, 2, 300)
    
    if recursion:
        tailleBande = findBandWith(x, bandwidths, 2)
    else:
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        #cv=LeaveOneOut()
                        #cv=is 3-Fold validation (default)
                        )
        grid.fit(x);
        tailleBande = grid.best_params_
        
        plots.plot_Grid_KDE(grid,bandwidths)

    print("taille bande is  = ", tailleBande['bandwidth'],"\n")
    #tailleBande = {'bandwidth':0.01}
    # instantiate and fit the KDE model
    kde = KernelDensity(
        bandwidth=tailleBande['bandwidth'],
       kernel='gaussian') 
    kde.fit(x);


    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x)
    LLH_tr = np.transpose([logprob]) #transpose
    std_scale = preprocessing.StandardScaler().fit(LLH_tr) #centrer reduit
    LLH_tr = std_scale.transform(LLH_tr)
    return kde
   # return np.transpose(LLH_tr)[0]
    






def removeOutliers(tabIn):
    tab = np.transpose(tabIn)
    listdf = [pandas.DataFrame(x) for x in tabIn]
    zscore = (np.abs(stats.zscore(listdf[0])))
    boolvar = [(np.abs(stats.zscore(df)) < 2.5).all(axis=1) for df in listdf]
    
    boolvar = np.transpose(boolvar)
    listdf2 = np.transpose(listdf)[0]

    tabOut = [np.mean([a for a,b in zip(x,y) if b]) for x,y  in zip(listdf2, boolvar)] 
    return np.array([tabOut]);
#    df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

###
def centreReduit(x):
    std_scale = preprocessing.StandardScaler().fit(x)
    return std_scale.transform(x) #centré réduit

