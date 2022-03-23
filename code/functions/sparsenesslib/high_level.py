#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#[EN]high level functions that organize the sequence of the other functions of the module, they would probably be difficult to reuse for another project

#[FR]fonctions de haut niveau qui organisent l'enchainement des autres fonctions du module, elles seraient probablement difficilement réutilisables pour un autre projet

#1. compute_sparseness_metrics_activations: compute metrics of the layers given in the list *layers* of the images contained in the directory *path*
    #by one of those 3 modes: flatten channel or filter (cf activations_structures subpackage) and store them in the dictionary *dict_output*.

#2. write_file: Writes the results of the performed analyses and their metadata in a structured csv file with 
    # a header line, 
    # results (one line per layer), 
    # a line with some '###', 
    # metadata

#3. layers_analysis: something like a main, but in a function (with all previous function),also, load paths, models/weights parameters and write log file

#####################################################################################
# LIBRAIRIES:
#####################################################################################
#public librairies
import time
import os
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
#from keras_vggface.vggface import VGGFace
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import sys
import statistics as st
from scipy import stats
from datetime import date
import pandas
import matplotlib.pyplot as plt
import numpy as np
import json
import re
import csv
#import vggplaces.vgg16_places_365 as places
#personnal librairies
sys.path.insert(1,'../../code/functions')
import sparsenesslib.keract as keract
import sparsenesslib.metrics as metrics
import sparsenesslib.plots as plots

import sparsenesslib.sparsenessmod as spm
import sparsenesslib.activations_structures as acst
#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################


def getPaths(bdd, pathData):
        #path d'enregistrement des résultats

    if pathData == 'sonia': #databases aren't in repo bc they need to be in DATA partition of the pc (more space)
        if bdd in ['CFD','JEN','SCUT-FBP','MART']:
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/'+bdd+'/labels_'+bdd+'.csv'
            log_path ='../../results/'+bdd+'/log_'
        elif bdd == 'SMALLTEST':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/small_test/labels_test.csv'
            log_path ='../../results/smalltest/log_'
        elif bdd == 'BIGTEST':        
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/big_test/labels_bigtest.csv'
            log_path ='../../results/bigtest/log_'

    else: #for others configurations,all paths are relative paths in git repository
        if pathData == 'LINUX-ES03':
            pathData = '../../'

        if bdd in ['CFD','JEN','SCUT-FBP','MART']:
            labels_path =pathData+'data/redesigned/'+bdd+'/labels_'+bdd+'.csv'
            images_path =pathData+'data/redesigned/'+bdd+'/images'
            log_path =pathData+'results/'+bdd+'/log_'
        elif bdd == 'SMALLTEST':
            labels_path =pathData+'data/redesigned/small_test/labels_test.csv'
            images_path =pathData+'data/redesigned/small_test/images'
            log_path =pathData+'results/smalltest/log_'            
        elif bdd == 'BIGTEST':
            labels_path =pathData+'data/redesigned/big_test/labels_bigtest.csv'
            images_path =pathData+'data/redesigned/big_test/images'
            log_path =pathData+'results/bigtest/log_'  
    return labels_path, images_path, log_path

def configModel(model_name, weight):
    if model_name == 'VGG16':
        if weight == 'imagenet':
            model = VGG16(weights = 'imagenet')
            layers = ['input_1','block1_conv1','block1_conv2','block1_pool','block2_conv1', 'block2_conv2','block2_pool',
            'block3_conv1','block3_conv2','block3_conv3','block3_pool','block4_conv1','block4_conv2','block4_conv3',
            'block4_pool', 'block5_conv1','block5_conv2','block5_conv3','block5_pool','flatten','fc1', 'fc2']
            flatten_layers = ['fc1','fc2','flatten']
        elif weight == 'vggface':
            model = VGGFace(model = 'vgg16', weights = 'vggface')
            layers = ['input_1','conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2','conv3_3',
            'pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5','flatten',
            'fc6/relu','fc7/relu']
            flatten_layers = ['flatten','fc6','fc6/relu','fc7','fc7/relu','fc8','fc8/softmax']
        elif weight == 'vggplaces':
            model = places.VGG16_Places365(weights='places')
            layers = ['input_1','block1_conv1','block1_conv2','block1_pool','block2_conv1', 'block2_conv2','block2_pool',
            'block3_conv1','block3_conv2','block3_conv3','block3_pool','block4_conv1','block4_conv2','block4_conv3',
            'block4_pool', 'block5_conv1','block5_conv2','block5_conv3','block5_pool','flatten','fc1', 'fc2']
            flatten_layers = ['fc1','fc2','flatten']
    elif model_name == 'resnet50':
        if weight == 'imagenet': 
            print('error, model not configured')
        elif weight == 'vggfaces':
            print('error, model not configured')  
    return model, layers, flatten_layers

def preprocess_Image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    image = img_to_array(img)      
    img = image.reshape(
        (1, image.shape[0], image.shape[1], image.shape[2]))   
    image = preprocess_input(img)
    return image

def compute_sparseness_metrics_activations(model, flatten_layers, path, dict_output, layers, computation, formula, freqmod,k):
    '''
    compute metrics of the layers given in the list *layers*
    of the images contained in the directory *path*
    by one of those 3 modes: flatten channel or filter (cf activations_structures subpackage)
    and store them in the dictionary *dict_output*.
    '''
    imgs = [f for f in os.listdir(path)]    
    
    for i, each in enumerate(imgs,start=1):

        if i%freqmod == 0:
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)
      
        img_path = path + "/" + each
        image = preprocess_Image(img_path)

        # récupération des activations
        activations = keract.get_activations(model, image)
        activations_dict = {}
        acst.compute_activations(layers, flatten_layers, computation, activations, activations_dict,formula,k)
        dict_output[each] = activations_dict
#####################################################################################


def getActivations_for_all_image(model,path, computation, formula, freqmod):
    '''! Retourne un dictionnaire par image des activations  
    
    '''
    print("the path is: ", path)
    imageActivation = {}
    imgs = [f for f in os.listdir(path)]
    for i, each in enumerate(imgs,  start=1):
        if i%freqmod == 0:         
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)

        
        img_path = path + "/" + each
        image = preprocess_Image(img_path)
        
        # récupération des activations
        activations = keract.get_activations(model, image)
        
        imageActivation[each] = activations
    return imageActivation


def get_activation_by_layer(activations,path,dict_output, formula, k, layer):
    """! a partir du dictionnaire de toutes les activations, extrait la layer choisis

    """
    for i, each in enumerate([f for f in os.listdir(path)],  start=1) :
        activations_dict = {}
        
        acst.compute_flatten(activations[each], activations_dict, layer, formula,k)
        
        dict_output[each] = activations_dict

def parse_activations_by_layer(model,path, dict_output, layer, computation, formula, freqmod,k):
    
    '''
    une fonction qui pour la couche et seulement la couche, stocke les activations de toutes les images
    elle retourne l'array des activations à la couche choisie    
    '''
    
    imgs = [f for f in os.listdir(path)] 
  
    
    for i, each in enumerate(imgs,start=1):
        if i%freqmod == 0:         
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)
       
        
        img_path = path + "/" + each
        image = preprocess_Image(img_path)
        
        # récupération des activations
        activations = keract.get_activations(model, image)
        
        activations_dict = {}
        
        acst.compute_flatten(activations, activations_dict, layer, formula,k)
        
        dict_output[each] = activations_dict
#####################################################################################
def parse_activations_by_filter(model,path, list_output, layer, computation, formula, freqmod,k):
    
    '''
    Une fonction qui pour la couche et seulement la couche, stocke les activations de toutes les images, par filtre
    Elle retourne dans list_output une liste (de taile n = nombre de filtres) de dictionnaires(de taille n= nombre d'images)
    d'array des activations par filtre a la couche choisie
    '''
    
    imgs = [f for f in os.listdir(path)] 
      

    #pour avoir le nombre d'activations, test sur une image quelconque
    img_path = path + "/" + imgs[1]
    image = preprocess_Image(img_path)


    activations = keract.get_activations(model, image)
    print(layer)
    #print(activations.items())
    [print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]


    for i, each in enumerate(imgs,start=1):
        

        if i%freqmod == 0:
            
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)
       
        
        img_path = path + "/" + each
        image = preprocess_Image(img_path)
        
        # récupération des activations
        activations = keract.get_activations(model, image)
        
        activations_dict = {}
        
        acst.compute_flatten(activations, activations_dict, layer, formula,k)  
        
        dict_output[each] = activations_dict
        
#####################################################################################
def write_file(log_path, bdd, weight, metric, df_metrics, df_reglog, df_scope, df_inflexions, layers, k):    
    '''
    Writes the results of the performed analyses and their metadata in a structured csv file with 
    - a header line, 
    - results (one line per layer), 
    - a line with some '###', 
    - metadata
    '''

    today = date.today()
    today = str(today)
    for i in range(2,31):
        df_metrics = df_metrics.rename(columns = {'input_'+i: 'input_1'})

    with open(log_path +'_'+bdd+'_'+weight+'_'+metric+'_'+today+'_ANALYSE'+'.csv',"w") as file:            
        #HEADER
        file.write('layer'+';'+'mean_'+str(metric)+';'+'sd_'+str(metric)+';'+'corr_beauty_VS_'+'metric'+';'+'pvalue'+';'+'\n')
        #VALUES for each layer
        for layer in layers:

            '''
            if layer[0:5] == 'input':
                layer = 'input' + '_' + str(k)'''
            file.write(layer+';')            
            #mean
            l1 = list(df_metrics[layer])
            file.write(str(st.mean(l1))+';')               
            #standard deviation
            l1 = list(df_metrics[layer])
            file.write(str(st.stdev(l1))+';')            
            #correlation with beauty
            l1 = list(df_metrics[layer])
            l2 = list(df_metrics['rate']) 
            reg = linregress(l1,l2)
            r = str(reg.rvalue)         
            file.write(r +';')             
            #pvalue
            pvalue = str(reg.pvalue) 
            file.write(pvalue+';'+'\n')   
        
        #METADATA
        file.write('##############'+'\n')        
        file.write('bdd;'+ bdd + '\n')        
        file.write('weights;'+ weight + '\n')         
        file.write('metric;'+ metric + '\n')            
        file.write("date:;"+today+'\n')
        #correlation with scope
        l1 = list(df_scope['reglog'])        
        l2 = list(df_scope['rate'])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue) 
        pvalue = str(reg.pvalue)
        file.write('coeff_scope: ;'+coeff+';pvalue:'+pvalue +'\n') 
        #correlation with coeff of logistic regression
        l1 = list(df_reglog['reglog'])        
        l2 = list(df_reglog['rate'])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue)
        pvalue = str(reg.pvalue)    
        file.write('coeff_reglog: ;'+coeff+';pvalue:'+pvalue +'\n')  
        #correlation with each inflexions points        
        l1 = list(df_inflexions['reglog'])        
        l2 = list(df_inflexions['rate'])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue) 
        pvalue = str(reg.pvalue)       
        file.write('coeff_slope_inflexion: ;'+coeff+';pvalue:'+pvalue +'\n') 
        
#####################################################################################
def extract_metrics(bdd,weight,metric, model_name, computer, freqmod,k = 1):
    '''
    something like a main, but in a function (with all previous function)
    ,also, load paths, models/weights parameters and write log file

    *k:index of the loop, default is 1*'''

    t0 = time.time()

    
    labels_path, images_path, log_path = getPaths(bdd, computer)
    model, layers, flatten_layers =configModel(model_name, weight)

    dict_compute_metric = {}    
    dict_labels = {}

    if metric == 'L0':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod,k)
    if metric == 'kurtosis':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod,k)
    if metric == 'L1':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod,k)
    if metric == 'hoyer':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod,k)
    if metric == 'gini_flatten':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', 'gini', freqmod, k)
    if metric == 'gini_channel':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'channel', 'gini', freqmod, k)
    if metric == 'gini_filter':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'filter', 'gini', freqmod, k)
    if metric == 'mean':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod, k)
    if metric == 'acp':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod, k)

    
    spm.parse_rates(labels_path, dict_labels)
    
    today = date.today()
    today = str(today)

    df_metrics = spm.create_dataframe(dict_labels, dict_compute_metric) 
    today = date.today()
    today = str(today)
    df_metrics.to_json(path_or_buf = log_path+'_'+bdd+'_'+weight+'_'+metric+'_'+'_BRUTMETRICS'+'.csv')
#####################################################################################
def extract_pc_acp(bdd,weight,metric, model_name, computer, freqmod,k = 1):
    '''
    something like a main, but in a function (with all previous function)
    ,also, load paths, models/weights parameters and write log file

    *k:index of the loop, default is 1*

    Version for compute pca (loop on layers before loop on pictures)    
    '''
    if computer == 'LINUX-ES03':
        computer = '../../'

    t0 = time.time()

    labels_path, images_path, log_path = getPaths(bdd, computer)
    model, layers, flatten_layers =configModel(model_name, weight)

    dict_compute_pc = {}   #un dictionnaire qui par couche, a ses composantes principales (et les coorodnnées de chaque image pour chaque composante)
    dict_labels = {}

    activations = getActivations_for_all_image(model,images_path,'flatten', metric, freqmod)
    

    for layer in layers:   

        
        print('##### current layer is: ', layer)
        #une fonction qui pour la couche et seulement la couche, stocke les activations de toutes les images
        #elle retourne l'array des activations à la couche choisie
        dict_activations = {}
        get_activation_by_layer(activations,images_path,dict_activations, metric, k, layer)
        
        #parse_activations_by_layer(model,images_path,dict_activations, layer, 'flatten', metric, freqmod, k)
        
        pc = []
        #une fonction qui fait une acp la dessus, qui prends en entrée la liste pc vide et l'array des activations,
        #et enregistre les coordonnées des individus pour chaque composante dans un csv dans results/bdd/pca
        metrics.acp_layers(dict_activations, pc, bdd, layer,False, computer)
        
    
    spm.parse_rates(labels_path, dict_labels)
    
    today = date.today()
    today = str(today)

def extract_pc_acp_block(bdd,weight,metric, model_name, computer, freqmod,k = 1):
    '''
    something like a main, but in a function (with all previous function)
    ,also, load paths, models/weights parameters and write log file

    *k:index of the loop, default is 1*

    Version for compute pca (loop on layers before loop on pictures) 
    '''

    t0 = time.time()
    labels_path, images_path, log_path = getPaths(bdd, computer)
    model, layers, flatten_layers =configModel(model_name, weight)

    dict_compute_pc = {}   #un dictionnaire qui par couche, a ses composantes principales (et les coorodnnées de chaque image pour chaque composante)
    dict_labels = {}

    lastBlock = ""
    for layer in layers:   
        block =""
        x = re.search("block\d*",layer)
        if( x==None ):
            block = layer
        else:
            block = x.group()
        if lastBlock != block:
            lastBlock = block
            print('##### current block is: ', block)
            #une fonction qui pour la couche et seulement la couche, stocke les activations de toutes les images
            #elle retourne l'array des activations à la couche choisie
            dict_activations = {}
        
            parse_activations_by_layer(model,images_path,dict_activations, block, 'flatten', metric, freqmod, k)

            pc = []
            #une fonction qui fait une acp la dessus, qui prends en entrée la liste pc vide et l'array des activations,
            #et enregistre les coordonnées des individus pour chaque composante dans un csv dans results/bdd/pca
            metrics.acp_layers(dict_activations, pc, bdd, block, True, computer)
    spm.parse_rates(labels_path, dict_labels)
    today = date.today()
    today = str(today)




#####################################################################################################""
def extract_pc_acp_filter(bdd,weight,metric, model_name, computer, freqmod,k = 1):
    '''
    something like a main, but in a function (with all previous function)
    ,also, load paths, models/weights parameters and write log file

    *k:index of the loop, default is 1*

    Version for compute pca (loop on layers before loop on pictures)
    
    '''

    t0 = time.time()

 
    labels_path, images_path, log_path =getPaths(bdd, computer)
    model, layers, flatten_layers =configModel(model_name, weight)

    dict_compute_pc = {}   #un dictionnaire qui par couche, a ses composantes principales (et les coorodnnées de chaque image pour chaque composante)
    dict_labels = {}

    print(layers)

    for layer in layers:   

        
        print('##### current layer is: ', layer)
        #une fonction qui pour la couche et seulement la couche, stocke les activations de toutes les images par filtre
        #elle retourne une liste (n = nombre de filtres) de dict (n = nombre d'images) d'arrays des activations (n = nombre d'activations du filtre)
        # à la couche choisie, de taille n avec n = nombre de filtres
        
        list_activations = {}
        
        
        parse_activations_by_filter(model,images_path,list_activations, layer, 'flatten', metric, freqmod, k)
        
        pc = []
        #une fonction qui fait une acp la dessus, qui prends en entrée la liste pc vide et l'array des activations,
        #  et retourne la liste remplie
        metrics.acp_layers(dict_activations, pc)
        
        #A CODER 

        #dict_compute_pc[layer] = pc
        
    
    spm.parse_rates(labels_path, dict_labels)
    
    today = date.today()
    today = str(today)

    

    
#####################################################################################
def analyse_metrics(model_name, computer, bdd, weight, metric,k):
    
    #récupération du nom des couches
    model, layers, flatten_layers =configModel(model_name, weight)

    labels_path, images_path, log_path = getPaths(bdd, computer)

    #chargement des noms des images
    dict_labels = {}
    spm.parse_rates(labels_path, dict_labels)
    #chargement des données  
    data = json.load(open(log_path+'_'+bdd+'_'+weight+'_'+metric+'_'+'_BRUTMETRICS'+'.csv', "r"))    
    df_metrics = pandas.DataFrame.from_dict(data)    
    #df_metrics = pandas.read_json(path_or_buf= log_path+'_'+bdd+'_'+weight+'_'+metric+'_'+'_BRUTMETRICS'+'.csv')
    #écriture des histogrammes      
    #metrics.histplot_metrics(layers, df_metrics, bdd, weight, metric, log_path,k)    
    #régression logistique   
    if metric in ['kurtosis','L0','mean']:              
        df_metrics = metrics.compress_metric(df_metrics, metric)  
        
        
    df_reglog = metrics.reglog(layers, df_metrics, dict_labels) 
    #minimum - maximum     
    df_scope = metrics.minmax(df_metrics,dict_labels)    
    #Gompertz    
    '''df_gompertz = metrics.reg_gompertz()'''
    #inflexion
    df_inflexions = metrics.inflexion_points(df_metrics,dict_labels)
    df_inflexions.to_json(path_or_buf = log_path+'_'+bdd+'_'+weight+'_'+metric+'_'+'_inflexions'+'.csv')
    #écriture du fichier

    write_file(log_path, bdd, weight, metric, df_metrics, df_reglog, df_scope, df_inflexions ,layers, k)    
#####################################################################################



def getAllFile(path, formatOrdre = []):
    files = []
    if len(formatOrdre)==0: #ordre de parcours alphabétique
        files = [f for f in os.listdir(path)]    
    else: #parcours les fichier qui match avec formatOrdre
        #if isinstance(formatOrdre[0], list):
            files = [formatOrdre[0]+f+formatOrdre[2] for f in formatOrdre[1]]
    return files


def eachFilePlot(path, formatOrdre = []):
    files =  getAllFile(path, formatOrdre)
    for each in files:
        csv_path = path + "/" + each
        if os.path.isfile(csv_path):
            
            x, head = readCsv(csv_path) #recupère le CSV
            
            print('######', each,"     ")
            plots.plotHist_fromFiles(x, head, name =each )

def eachFile(path, formatOrdre = []):
    files =  getAllFile(path, formatOrdre)
    tabcsv =[]
    for each in files:
        csv_path = path + "/" + each
        if os.path.isfile(csv_path):
            
            x, head = readCsv(csv_path) #recupère le CSV
            tabcsv.append(x)
    return tabcsv


def eachFileCSV(path, formatOrdre = [], pathForLLH=[]):
    """! parcours tout les chifier du repertoire path, fait: mixureGaussian, LLH, tableau des nbe de PC par couche
    
    @param path chemin des CSV a traiter
    @param formatOrdre permet de parcourir dans un ordre précis:
        syntaxe: formatOrdre[  prefixe, TabName[], sufixe]
    @param pathForLLH path pour les log likeliHood

    @return tableau des nbe de PC par couche
    """
    tabPC = []
    pathPCA = path+"/"+"pca"
    pathHist = path+"/"+"histo"

    files = getAllFile(pathPCA, formatOrdre)

    for each in files:
        csv_path = pathPCA + "/" + each
        x, _ = readCsv(csv_path) #recupère le CSV
        tabPC.append(x.shape[1])

        print('######', each,"     ", x.shape[1])
        gm = metrics.getMultigaussian(x,name =  pathPCA+" "+each, plot=[True,False], nbMaxComp =10)
        
       # metrics.doVarianceOfGMM(gm, x)
        allLLH =  metrics.DoMultipleLLH(gm, x,100)
        allLLH =metrics.removeOutliers(allLLH)
        allHist, legend = metrics.doHist(allLLH, True, "distributions des LLH pour GMM")
        #metrics.writeHist(allHist, legend,pathHist,"_nbComp="+str(gm.n_components)+"_covarType="+gm.covariance_type+"_"+each)
        """
            if len(pathForLLH)>0:
                if len(pathForLLH)>2:
                    pathForLLH[2] = (each)
                else: pathForLLH.append(each)
                metrics.getlogLikelihood(gm, x, pathForLLH,  True)
        """
    return tabPC
        
def eachFileCSV_Centroid(path, formatOrdre = []):
    """!
    
    @param path[] repertoire des CSV a traiter avec repertoire variance
    @param formatOrdre permet de parcourir dans un ordre précis:
        syntaxe: formatOrdre[  prefixe[], TabName[], sufixe]

    """
    #getAllFile(path[0], [formatOrdre[0][0], formatOrdre[1],formatOrdre[2]] )

    if len(formatOrdre)==0: #ordre de parcours alphabétique
        filesCP = [f for f in os.listdir(path[0])]    
        filesVar = [f for f in os.listdir(path[1])]    
    else: #parcours les fichier qui match avec formatOrdre
        filesCP = [formatOrdre[0][0]+f+formatOrdre[2] for f in formatOrdre[1]]
        filesVar = [formatOrdre[0][1]+f+formatOrdre[2] for f in formatOrdre[1]]  
   
    for eachCP,eachVar in zip(filesCP, filesVar):
        csv_pathCP = path[0] + "/" + eachCP
        csv_pathVar = path[1] + "/" + eachVar
        cp, _ = readCsv(csv_pathCP) #recupère le CSV
        var, _ = readCsv(csv_pathVar) #recupère le CSV
        

        print('######', eachCP,"  ",eachVar,"   ", cp.shape[1])
        metrics.distToCentroid(cp, var, eachCP+"\n distance du Centroïd")

def eachFileCSV_Kernel(path, filesPC):
    """!
    """
    #tabPC = []
    pathPCA = path+"/"+"pca"
    
    #pathHist = path+"/"+"histo"

    #files = getAllFile(pathPCA, formatOrdre)
   
    for each in filesPC:
        x, _ = readCsv(pathPCA + "/" + each)  #recupère le CSV
        kde= metrics.KDE(x)

        AllLLH =  metrics.DoMultipleLLH(kde, x,1)

        metrics.doHist(AllLLH, plot = True, name = "distributions des LLH pour KDE")


def each_compare_GMM_KDE(path, filesPC):
    """!
    """
    #tabPC = []
    pathPCA = path+"/"+"pca"
    
    #pathHist = path+"/"+"histo"

    #files = getAllFile(pathPCA, formatOrdre)
    AllSpearman = []
    for each in filesPC:
        x, _ = readCsv(pathPCA + "/" + each)  #recupère le CSV
        kde = metrics.KDE(x)
        LLH_KDE =  metrics.DoMultipleLLH(kde, x,1)[0]
#        gm = metrics.getMultigaussian(x,name =  pathPCA+" "+each, plot=[True,False], nbMaxComp =10)
        gm = metrics.getMultigaussian(x,name =  pathPCA+" "+each, plot=[False,False], nbMaxComp =10)
        
       # metrics.doVarianceOfGMM(gm, x)
        LLH_GMM =  metrics.DoMultipleLLH(gm, x,50)
        LLH_GMM =metrics.removeOutliers(LLH_GMM)
        
        AllSpearman.append( metrics.spearman(LLH_KDE, LLH_GMM[0]))

       # plots.plotHist(np.array([LLH_KDE,LLH_GMM[0]]), name= "distribution des LLH\n KDE         |           GMM", max = 2)
       # metrics.compareValue(LLH_KDE, LLH_GMM[0], "Difference LLH entre GMM et KDE")
        #metrics.CompareOrdre(LLH_KDE, LLH_GMM[0], "Difference d'ordre LLH entre GMM et KDE")
        #metrics.doHist(np.array([LLH]), plot = True)
    print( AllSpearman)
    writeCSV=True
    if writeCSV:
        pathSpearman = path+"/"+"spearman"
        df = pandas.DataFrame(AllSpearman)
        df = df.transpose()
        df.columns = filesPC
        df = df.transpose()
        #df = df.transpose()
        #df.columns = legend[0:-1]
        #os.makedirs(pathData+"results"+"/"+bdd+"/"+"LLH", exist_ok=True)
            #l'enregistrer dans results, en précisant la layer dans le nom
        os.makedirs(pathSpearman, exist_ok=True)
        df.to_csv(pathSpearman+"/corr_spearman.csv")
    return AllSpearman


def readCsv(path):
    """
    lit un fichier .csv, convertit toutes les valeurs en float et retourne un numpyArray
    """
    try:
        with open(path, newline='') as csvfile:
            rows = list(csv.reader(csvfile,delimiter=','))
            #rows[0].pop(0)
            for row in rows[0:]:
                for i in range(len(row)):
                    try:
                        row[i] = float(row[i])
                    except:
                        pass
                row.pop(0)
            head = rows.pop(0)
            return np.array(rows), head
    except OSError:
        print("cannot open", path)
        return None
    else:
        print("an error has occurred ")
        return None


