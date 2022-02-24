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


def BIC(X, verbose = False, plot = False):
    """! Bayesian Information Criterion
    """
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            if verbose:
                verbosePourcent(n_components*(cv_types.index(cv_type)),len(n_components_range)*len(cv_types) )
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    
    if plot:
        plots.plotBIC(bic, best_gmm, n_components_range = n_components_range)

    return best_gmm
   # clf = best_gmm
    #bars = []





def getMultigaussian(X, name ="Gaussian Mixture", index = 1):
    
    X_scale = X #MultiDimensionalScaling(X)
    
    
    # Centrage et Réduction
   # std_scale = preprocessing.StandardScaler().fit(X_transform) 
    #X_scale = std_scale.transform(X_transform)
    #SilouhetteCoef(X_scale, showGraphe=True, verbose = True)
    gm = BIC(X_scale, verbose = True, plot = True)

    gm2 = GaussianMixture(n_components =3, n_init = 2).fit(X_scale)
    #gm = BayesianGaussianMixture(n_components =10, n_init = 2, weight_concentration_prior_type ="dirichlet_process", weight_concentration_prior =0.0000000001).fit(X_scale)
    #print(gm.means);
    #print(gm)
    #plot_results(X_scale, gm.predict(X_scale), gm.means_, gm.covariances_, 0, "Gaussian Mixture")
    #plots.plot_MultiGaussian(X_scale, gm2, index, name)
    plots.plot_MultiGaussian(X_scale, gm, index, name)
  #  plot_gmm(gm, X_scale)














import scipy.stats

def logLikelihood(data, x = None):
    
    #if x==None:
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
    
    """    
    else:
    
    
    y=[]
    y.append(scipy.stats.norm.logpdf(data,x,0.5).sum())
    plt.plot(x,y)
    plt.title(r'Log-Likelihood')
    plt.xlabel(r'$\mu$')

    plt.grid()

    #plt.savefig("likelihood_normal_distribution_02.png", bbox_inches='tight')
    plt.show()
    """















def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
#        width, height = 2.0 * np.sqrt(2.0) * np.sqrt( linalg.eigh(covariance) )
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.title("GMM with %d components"%len(gmm.means_), fontsize=(20))
    plt.xlabel("U.A.")
    plt.ylabel("U.A.")
    plt.show()