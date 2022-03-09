#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#[EN]These functions allow to compute metrics on activation values correlating with sparsity (Gini index, L0 norm etc) 
#and the "metametrics" allowing to characterize them (regressions on the curve of the metric according to the layer, distributions...)

#[FR]Ces fonctions permettent de calculer les métriques sur les valeurs d'activation corrélant avec la sparsité (indice de Gini, norme L0 etc) 
#ainsi que les méta métriques permettant de les caractériser (régressions sur la courbe de la métriue en fonction de la couche, distributions ...)

#1. gini: Compute Gini coefficient of an iterable object (lis, np.array etc)

#2. treve-rolls: Compute modified treve-rolls population sparsennes, formula from (wilmore et all, 2000)

#3. reglog: Compute a logistic regression for each picture between layer"s metric value (y) and number of layer (x)

#4. minmax: Compute for each picture the difference between the higher and lower value of layer's metrics 

#5. gompertzFct: Define the Gompertz function model

#6. reg_gompertz: Compute a regression on the Gompertz function. Function for the moment not functional. 

#7. histplot_metrics: Plot a histogram of the distribution of metrics for all images regardless of the layer in which they were calculated

#8. compress_metrics: If metrics values aren't between 0 and 1, like kurtosis or L0 norm, change theme like this to let them availaible 
#for logistic regression. 1 will be th highest value of the metric, 0 the lowest.

#####################################################################################
# LIBRAIRIES:
#####################################################################################
#public librairies
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
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

from sklearn import decomposition
from sklearn.decomposition import IncrementalPCA
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn import metrics

import csv


###############
#Verbose Procedures
#permet d'afficher les pourcentage de progression
def verbosePourcent(current, valMax):
    if verbosePourcent.pourcent < int(current*100/valMax):
        print("\033[A                             \033[A")
        print( int(current*100/valMax) , "%" )
        verbosePourcent.pourcent = int(current*100/valMax)
    if verbosePourcent.pourcent ==100:
        verbosePourcent.pourcent = 0 # reset le compteur une fois 100% atteint
verbosePourcent.pourcent=0
##########


#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################
def hoyer(vector):
    '''
    Hoyer's measure of sparsity for a vector
    '''
    sqrt_n = np.sqrt(len(vector))
    return (sqrt_n - np.linalg.norm(vector, 1) / np.linalg.norm(vector)) / (sqrt_n - 1)
#####################################################################################
def gini(vector):
    '''
    Compute Gini coefficient of an iterable object (lis, np.array etc)
    '''    
    if np.amin(vector) < 0:
        # Values cannot be negative:
        vector -= np.amin(vector)
    # Values cannot be 0:     
    vector = [i+0.0000001 for i in vector]     
    # Values must be sorted:
    vector = np.sort(vector)
    # Index per array element:
    index = np.arange(1,vector.shape[0]+1)
    # Number of array elements:
    n = vector.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * vector)) / (n * np.sum(vector)))
#####################################################################################
def treves_rolls(vector):
    '''
    compute modified treve-rolls population sparsennes, formula from (wilmore et all, 2000)
    '''
    denominator = 0
    numerator = 0
    length = len(vector)
    for each in vector:
        numerator += abs(each)
        denominator += (each*each)/length 
    tr=1 - (((numerator/length)*(numerator/length)) /denominator)
    return tr 
#####################################################################################
def reglog(layers, df_metrics,dict_labels):
    '''
    compute a logistic regression for each picture between layer"s metric value (y) and number of layer (x)
    '''
    i = 1.0 
    x = []  
    for each in range(len(layers)):
        x.append(i)
        i += 1   
            
    x = pandas.DataFrame(x) 

    dict_reglog = {}

    for row in df_metrics.iterrows():
        y = []
        j = 0
        for each in list(row)[1]:            
            if j != 0:
                y.append(each)
            j += 1       

        picture = list(row)[0]

        y= pandas.DataFrame(y)
        df = pandas.concat([x,y], axis=1)   
        df.columns = ['x', 'y']
        # on ajoute une colonne pour la constante
        x_stat = sm.add_constant(x)
        # on ajuste le modèle
        model = sm.Logit(y, x_stat)
        result = model.fit(disp=0)    
        #on récupère le coefficient
        coeff = result.params[0]
        dict_reglog[picture] = coeff        

    df1 = pandas.DataFrame.from_dict(dict_labels, orient='index', columns = ['rate'])
    df2 = pandas.DataFrame.from_dict(dict_reglog, orient='index', columns = ['reglog']) 

    return pandas.concat([df1, df2], axis = 1)     
#####################################################################################
def minmax(df_metrics,dict_labels):
    '''
    compute for each picture the difference between the higher and lower value of layer's metrics 
    '''
    dict_scope = {}

    for row in df_metrics.iterrows():
        y = []
        j = 0
        
        for each in list(row)[1]:
            if j != 0:
                y.append(each)
            j += 1   

        picture = list(row)[0]

        maximum = max(y)
        minimum = min(y)
        diff = maximum - minimum

        dict_scope[picture] = diff     

    df1 = pandas.DataFrame.from_dict(dict_labels, orient='index', columns = ['rate'])
    df2 = pandas.DataFrame.from_dict(dict_scope, orient='index', columns = ['reglog'] ) 
    return pandas.concat([df1, df2], axis = 1)  
#####################################################################################
def gompertzFct (t , N , r , t0 ):
    '''
    Define the Gompertz function model
    '''
    return N * np . exp ( - np . exp ( - r * (t - t0 ))) 
#####################################################################################
def reg_gompertz(x,y, df_gompertz):
    '''
    Compute a regression on the Gompertz function. 
    Function for the moment not functional. 
    '''
    I_t = y [ x :]
    t = np.arange (len( I_t ))

    model = gompertzFct
    guess = (100000. , .1 , 50.)

    parameters , variances = opt . curve_fit ( model , t , I_t , p0 = guess )

    G_t = model (t , * parameters )

    print ( np . sqrt ( np . mean (( I_t - G_t )**2)))
#####################################################################################
def histplot_metrics(layers, df_metrics, bdd, weight, metric, log_path,k):
    '''
    Plot a histogram of the distribution of metrics for all images 
    regardless of the layer in which they were calculated
    '''    
    y = []    
    for layer in layers:
        if layer[0:5] == 'input':
            layer = 'input' + '_' + str(k)         
        y = list(itertools.chain(y, list(df_metrics[layer])))
    title = 'distrib_'+ bdd +'_'+ weight +'_'+ metric   
    plt.hist(y, bins = 40)        
    plt.title(title, fontsize=10)                 
    plt.savefig(log_path +'_'+ bdd +'_'+ weight +'_'+ metric +'.png')
    plt.clf()
#####################################################################################
def compress_metric(df_metrics, metric):
    '''
    If metrics values aren't between 0 and 1, like kurtosis or L0 norm, change theme like this
    to let them availaible for logistic regression. 1 will be th highest value of the metric, 0 the lowest.
    '''
    vmax = 0
    vmin = 0

    #suppression temporaire de la note
    rates = df_metrics.pop('rate')     

    #récupération de la valeur minimale
    j = 0
    for row in df_metrics.index:        
        i = 0
        for index_row, row in df_metrics.iterrows():                     
            for column in row:           
                if i != 0:                                
   
                    if column < vmin:
                        vmin = column                 
            i += 1
        j += 1
    
    #transformation des valeurs pour qu'elles soient strictement positives    
    df_metrics = df_metrics.applymap(lambda x: x + abs(vmin)) 

    #récupération de la valeur maximale
    j = 0
    for row in df_metrics.index:        
        i = 0
        for index_row, row in df_metrics.iterrows():                     
            for column in row:           
                if i != 0:                                
                    if column > vmax:
                        vmax = column                   
            i += 1
        j += 1   
     
    #transformation des valeurs pour qu'elles soient entre 0 et 1    
    df_metrics = df_metrics.applymap(lambda x: x/vmax) 

    #ajout de la note
    df_metrics.insert(0, 'rate', rates, allow_duplicates=False)    

    return df_metrics
#####################################################################################
def inflexion_points(df_metrics,dict_labels):
    '''
    from here: https://stackoverflow.com/questions/62537703/how-to-find-inflection-point-in-python
    '''
    dict_inflexions = {}

    for row in df_metrics.iterrows():
        y = []
        j = 0
        
        for each in list(row)[1]:
            if j != 0:
                y.append(each)
            j += 1   

        picture = list(row)[0]


        smooth = gaussian_filter1d(y, 3)

        # compute second derivative
        smooth_d2 = np.gradient(np.gradient(smooth))

        smooth_d1 = np.gradient(smooth)
        
        y_d1 = np.gradient(y)

        # find switching points
        infls = np.where(np.diff(np.sign(smooth_d2)))[0]

        coeff = y_d1[max(infls)]
        

        '''
        # plot results
        plt.plot(y, label='Noisy Data')
        plt.plot(smooth, label='Smoothed Data')


        plt.plot(smooth_d2 / np.max(smooth_d2), label='Second Derivative (scaled)')
        for i, infl in enumerate(infls, 1):
            plt.axvline(x=infl, color='k', label=f'Inflection Point {i}')
        plt.legend(bbox_to_anchor=(1.55, 1.0))  ''' 
           

        dict_inflexions[picture] = coeff     

        #print('infs: ', infls,'longueur: ',len(infls) )

    df1 = pandas.DataFrame.from_dict(dict_labels, orient='index', columns = ['rate'])
    df2 = pandas.DataFrame.from_dict(dict_inflexions, orient='index', columns = ['reglog'] ) 
    return pandas.concat([df1, df2], axis = 1)  
#####################################################################################
def acp_layers(dict_metrics, pc, bdd, layer, block = False, pathData = "../../"):
    
    '''
    A PCA with activations of each layer as features
    '''
    
    #conversion d'un dictionnaire avec chaque image en clé et un vecteur de toutes leurs activations en valeur, en pandas dataframe
    df_metrics = pandas.DataFrame.from_dict(dict_metrics)     
      
    tic = time.perf_counter()    

    for index, row in df_metrics.iterrows():         
        
        print('a') #flags pour monitorer visuellement le temps d'exécution de chaque étape (en fonction des jeux de données c'est pas au même endroit que ça plante)
        n_comp = 10 #nombre de composantes à calculer, fixé de manière à ce que leur somme soit au moins supérieure à 35  (a passer en paramètres)
        print('b')
        df = pandas.DataFrame.from_dict(dict(zip(row.index, row.values))).T  
        X = df.values 
        print('d')    
        # Centrage et Réduction
        std_scale = preprocessing.StandardScaler().fit(X)
        print('e')        
        X_scaled = std_scale.transform(X)
        print('f')        
        # Calcul des composantes principales        
        pca = decomposition.PCA(n_components= 0.8)

        print("g")     
        coordinates = pca.fit_transform(X_scaled)      
        print("h") 
        df = pandas.DataFrame(coordinates)
        print("i")
        bdd = bdd.lower()
        if block:
            os.makedirs(pathData+"results"+"/"+bdd+"/"+"pcaBlock", exist_ok=True)
            #l'enregistrer dans results, en précisant la layer dans le nom
            df.to_csv(pathData+"results"+"/"+bdd+"/"+"pcaBlock"+"/"+"pca_values_"+layer+".csv")
        else:
            os.makedirs(pathData+"results"+"/"+bdd+"/"+"pca", exist_ok=True)
            #l'enregistrer dans results, en précisant la layer dans le nom
            df.to_csv(pathData+"results"+"/"+bdd+"/"+"pca"+"/"+"pca_values_"+layer+".csv")

        #timer pour l'ACP de chaque couche
        print('############################################################################')
        toc = time.perf_counter()
        print(f"time: {toc - tic:0.4f} seconds")
        print('############################################################################')

        getVarienceRatio(pca,bdd, layer, pathData)

def getVarienceRatio(pca, bdd, layer, pathData = "../../"):
   
    variance = pca.explained_variance_ratio_ #calculate variance ratios

    var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
    print( var) #cumulative sum of variance explained with [n] features
    df = pandas.DataFrame(variance).transpose()
    df2 = pandas.DataFrame(var).transpose()
    os.makedirs(pathData+"results"+"/"+bdd+"/"+"pca_variance", exist_ok=True)
            #l'enregistrer dans results, en précisant la layer dans le nom
    df.to_csv(pathData+"results"+"/"+bdd+"/"+"pca_variance"+"/"+"variance"+layer+".csv")
    df2.to_csv(pathData+"results"+"/"+bdd+"/"+"pca_variance"+"/"+"varianceCumule_"+layer+".csv")


def Acp_densiteProba(dict_metrics, pc, bdd, layer):
    '''
    A PCA with activations of each bloc as features
    '''
    #conversion d'un dictionnaire avec chaque image en clé et un vecteur de toutes leurs activations en valeur, en pandas dataframe
    df_metrics = pandas.DataFrame.from_dict(dict_metrics)     
      
    tic = time.perf_counter()    



def readCsv(path):
    """
    lit un fichier .csv, convertit toutes les valeurs en float et retourne un numpyArray
    """
    try:
        with open(path, newline='') as csvfile:
            rows = list(csv.reader(csvfile,delimiter=','))
            for row in rows[1:]:
                for i in range(len(row)):
                    row[i] = float(row[i])
                row.pop(0)
            rows.pop(0)
            return np.array(rows)
    except OSError:
        print("cannot open", path)
        return None
    else:
        print("an error has occurred ")
        return None




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
        plots.plot_MultiGaussian(X, gm, index, name, X_MDS)
    return gm




def getlogLikelihood(gm, X, path, writeCSV = True):
    """! récupère le log likelihood et l'écris dans un CSV si writeCSV = True
    @param gm mixure gaussian
    @param X array de dimension N
    @param tableau de 2, path + bdd
    @writeCSV boolean
    @return array de LLH
    """
    pathData,bdd, layer = path
    LLH = gm.score_samples(X); #Compute the log-likelihood of each sample.
    
    if writeCSV:
        df = pandas.DataFrame(LLH)
        df = df.transpose()
        
        os.makedirs(pathData+"results"+"/"+bdd+"/"+"LLH", exist_ok=True)
            #l'enregistrer dans results, en précisant la layer dans le nom
        df.to_csv(pathData+"results"+"/"+bdd+"/"+"LLH"+"/"+"LLH"+layer)
    return LLH



def DoMultipleLLH(gmm, X, nbe):
    AllLLH = []
    for i in range(nbe):
        gmm.fit(X)
        LLH = gmm.score_samples(X); #récupère LLH

        LLH_tr = np.transpose([LLH]) #transpose
        std_scale = preprocessing.StandardScaler().fit(LLH_tr) #centrer reduit
        LLH_tr = std_scale.transform(LLH_tr)

        AllLLH.append(np.transpose(LLH_tr)[0])

#    AllLLH = np.array(AllLLH)
    return np.array(AllLLH)

def doHist(AllLLH, plot = False):
    bin = np.linspace(AllLLH.min(), AllLLH.max(),200)
    allHist = []
    legend = []
    for i, llh in enumerate(AllLLH):
        if plot and i <10:
            #spl = plt.subplot(5, 2, 1+i)
            hist = plt.hist(llh, bins=bin)
            plt.grid()
        hist = np.histogram(llh, bins=bin)
        legend = np.transpose(hist)[1]
        allHist.append(np.transpose(hist)[0])
    
    if plot:
        plt.show()
    #compareValue(AllLLH[0], AllLLH[1])
    return allHist, legend


def writeHist(allHist, legend, path,name):
   
    df = pandas.DataFrame(allHist)
    #df = df.transpose()
    df.columns = legend[0:-1]
    #os.makedirs(pathData+"results"+"/"+bdd+"/"+"LLH", exist_ok=True)
        #l'enregistrer dans results, en précisant la layer dans le nom
    os.makedirs(path, exist_ok=True)
    df.to_csv(path+"/"+name)


def compareValue(X, Y):
    
    val = abs(X - Y)
    bin = np.linspace(val.min(), val.max(),100)
    hist = plt.hist(val, bins=bin)
    plt.grid()
    plt.show()


def doVarianceOfGMM(gmm, X, plot = False):
    AllLLH = DoMultipleLLH(gmm, X)

    allHist, legend = doHist(allLLH)

    writeCSV=True
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

def KDE(x):
    # Centrage et Réduction
    std_scale = preprocessing.StandardScaler().fit(x)
    x = std_scale.transform(x)

    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        #cv=LeaveOneOut()
                        #cv=is 5-Fold validation (default)
                        )
    grid.fit(x);
    tailleBande = grid.best_params_

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=tailleBande['bandwidth'], kernel='gaussian')
    kde.fit(x)

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x)
    LLH_tr = np.transpose([logprob]) #transpose
    std_scale = preprocessing.StandardScaler().fit(LLH_tr) #centrer reduit
    LLH_tr = std_scale.transform(LLH_tr)

    doHist(np.array([LLH_tr]), plot = True)
    #plt.fill_between(x, np.exp(logprob), alpha=0.5)
    #plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
    #plt.ylim(-0.02, 0.22)
    #plt.show()
    






#plt.plot(range(len(X)),value1)



import scipy.stats

#inutilisé pour le moment
def logLikelihood(data, x = None):

    if type(x) is np.ndarray:
        x = np.linspace(data.min(), data.max(), 1000, endpoint=True)
        y=[]
        for i in x:
            y.append(scipy.stats.norm.logpdf(data,i,0.5).sum())
        plt.plot(x,y)
        plt.title(r'Log-Likelihood')
        plt.xlabel(r'$\mu$')

        plt.grid()

        #plt.savefig("likelihood_normal_distribution_02.png", bbox_inches='tight')
        plt.show()
    



