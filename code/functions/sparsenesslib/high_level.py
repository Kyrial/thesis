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
from more_itertools import chunked
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
import sparsenesslib.metrics_melvin as metrics_melvin
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

        if bdd in ['CFD','JEN','SCUT-FBP','MART','CFD_1','CFD_AF','CFD_F','CFD_WM']:
            labels_path =pathData+'data/redesigned/'+bdd+'/labels_'+bdd+'.csv'
            images_path =pathData+'data/redesigned/'+bdd+'/images'
            log_path =pathData+'results/'+bdd+'/log_'
        elif bdd in ['CFD_ALL']:
            labels_path =pathData+'data/redesigned/CFD/labels_CFD.csv'
            images_path =pathData+'data/redesigned/CFD/images'
            log_path =pathData+'results/CFD/log_'
        elif bdd == 'SMALLTEST':
            labels_path =pathData+'data/redesigned/small_test/labels_test.csv'
            images_path =pathData+'data/redesigned/small_test/images'
            log_path =pathData+'results/smalltest/log_'            
        elif bdd == 'BIGTEST':
            labels_path =pathData+'data/redesigned/big_test/labels_bigtest.csv'
            images_path =pathData+'data/redesigned/big_test/images'
            log_path =pathData+'results/bigtest/log_'  
        elif bdd == 'Fairface':
            labels_path =pathData+'data/redesigned/Fairface/fairface_label_train.csv'
            images_path =pathData+'data/redesigned/Fairface/'
            log_path =pathData+'results/Fairface/log_'  
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


def getActivations_for_all_image(model,path, imgs, computation, formula, freqmod):
    '''! Retourne un dictionnaire par image des activations  
    
    '''

    


    #print("the path is: ", path)
    imageActivation = {}
    #imgs = [f for f in os.listdir(path)]
    for i, each in enumerate(imgs,  start=1):
        if i%freqmod == 0:         
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)

        
        img_path = path + "/" + each
        image = preprocess_Image(img_path)
        
        # récupération des activations
        activations = keract.get_activations(model, image)
        
        imageActivation[each] = activations
    return imageActivation


def get_activation_by_layer(activations,imgList,dict_output,computation, formula, k, layer):
    """! a partir du dictionnaire de toutes les activations, extrait la layer choisis

    """
    #for i, each in enumerate([f for f in os.listdir(path)],  start=1) :
    for i, each in enumerate(imgList,  start=1) :
        activations_dict = {}
        
        if computation == 'flatten' or layer in ['fc1','fc2','flatten']:
            if formula in ['mean', 'max']:
                formula = "acp"
            acst.compute_flatten(activations[each], activations_dict, layer, formula,k)
        elif computation == 'featureMap':
            acst.compute_flatten_byCarte(activations[each], activations_dict, layer, formula,k)
        else:
            print('ERROR: Computation setting isnt flatten or featureMap')
            return -1


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




def extract_pc_acp(bdd, layers, computation, freqmod, model, images_path,imglist, k, loadModele, metric, path, saveModele):
    '''
    something like a main, but in a function (with all previous function)
    ,also, load paths, models/weights parameters and write log file

    *k:index of the loop, default is 1*

    Version for compute pca (loop on layers before loop on pictures)    
    '''
    
    print("longueur imglist: ", len(imglist))
    activations = getActivations_for_all_image(model,images_path,imglist,computation, metric, freqmod)
    
    
    #nbComp = pandas.DataFrame()
    for layer in layers:
    
        
        print('##### current layer is: ', layer)
        #une fonction qui pour la couche et seulement la couche, stocke les activations de toutes les images
        #elle retourne l'array des activations à la couche choisie
        dict_activations = {}
        get_activation_by_layer(activations,imglist,dict_activations,computation, metric, k, layer)
        
        #parse_activations_by_layer(model,images_path,dict_activations, layer, 'flatten', metric, freqmod, k)
        
        pc = []
        #une fonction qui fait une acp la dessus, qui prends en entrée la liste pc vide et l'array des activations,
        #et enregistre les coordonnées des individus pour chaque composante dans un csv dans results/bdd/pca
        
    
        if computation == 'flatten' or layer in ['fc1','fc2','flatten']:
            if loadModele!="":
                comp = metrics.acp_layers_loadModele(dict_activations, pc, bdd, layer, path,modelePath = loadModele)
            else:
                comp = metrics.acp_layers(dict_activations, pc, bdd, layer, path,saveModele = saveModele)
        elif computation == 'featureMap':
            if loadModele!="":
                comp = metrics.acp_layers_featureMap_loadModele(dict_activations, pc, bdd, layer, path,modelePath = loadModele)
            else:
                comp = metrics.acp_layers_featureMap(dict_activations, pc, bdd, layer, path, saveModele = saveModele)



def preprocess_ACP(bdd,weight,metric, model_name, computer, freqmod,k = 1,computation = 'flatten',saveModele = False,loadModele=""):
    '''!Met en place les composants pour l'execution de l'ACP.   
        adapte les chemins d'entrée et sortie.
        prétraite certaines bases de donnée (CFD_ALL, fairface)
    '''


    ####### Passe le booleen a True pour CFD_ALL
         ## CFD_ALL pour découper tout en sous ensemble CFD
    allCFD = False
    if bdd == "CFD_ALL":
        allCFD = True
        bdd = "CFD"
    #######

    t0 = time.time()


    if computer == 'LINUX-ES03':
        computer = '../../'

    labels_path, images_path, log_path = getPaths(bdd, computer)
    model, layers, flatten_layers =configModel(model_name, weight)
    
    #sur le mesoLR, le chemin d'écriture et de lecture est différent
    if computer == '/home/tieos/work_cefe_swp-smp/melvin/thesis/':  #lecture
            computer = '/lustre/tieos/work_cefe_swp-smp/melvin/thesis/' #ecriture
    
    ## indique le chemin a charger si on charge le modèle        
    if loadModele !="":
        loadModele = computer+loadModele
    #adapte le chemin suivant la methode 
    if computation == 'flatten':
        path= computer+"results"+"/"+bdd+"/pca"
    elif computation == 'featureMap': 
        path= computer+"results"+"/"+bdd+"/FeatureMap"


    #dict_compute_pc = {}   #un dictionnaire qui par couche, a ses composantes principales (et les coorodnnées de chaque image pour chaque composante)
    #dict_labels = {}
    print("path :", computer)

   ####### lance l'ACP sur chaque sous ensemble de CFD
    if allCFD == True:
        combinaison = getAllGenreEthnieCFD(labels_path, exception= {'ethnie' : ["M","I"]})

        for key in combinaison.keys():
            if key == "":
                bdd = "CFD"
            else:
                bdd = "CFD_"+key
            imglist = combinaison[key]
            #adapte le chemin suivant la methode 
            if computation == 'flatten':
                path= computer+"results"+"/"+bdd+"/pca"
            elif computation == 'featureMap': 
                path= computer+"results"+"/"+bdd+"/FeatureMap"
            extract_pc_acp(bdd,layers, computation, freqmod,  model,images_path, imglist, k, loadModele, metric, path, saveModele)
    #######   
    else:
        if bdd == "Fairface":
            #filt = {'ethnie' : "Asian", 'genre' : "Female"}
            filt = {'ethnie' : "White", 'genre' : "Male"}
            imglist = parserFairface(labels_path,filt)
            #imglist = parserFairface(labels_path)
            for key, item in filt.items():
                if bdd == "Fairface":
                    bdd = bdd+"_"
                bdd = bdd+item[0]
            print("BDD: ", bdd,"\n\n")
        else:
            imglist = [f for f in os.listdir(images_path)]
    

        extract_pc_acp(bdd, layers, computation, freqmod,  model,images_path,imglist, k, loadModele, metric, path, saveModele)
        
    if '/lustre/tieos/work_cefe_swp-smp/melvin/thesis/' in path:
        path = '/home/tieos/work_cefe_swp-smp/melvin/thesis/'
    
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
        
    
    #spm.parse_rates(labels_path, dict_labels)
    
    today = date.today()
    today = str(today)

    
def preprocess_average(bdd,weight,metric, model_name, computer, freqmod,k = 1,computation = 'featureMap'):
    """!
    @param 

    @return 
    """
    def average():
        for count, batch in enumerate(list(chunked(imglist,100))):
            #if count < 104:
            #    continue
            print(count, batch)
            activations = getActivations_for_all_image(model,images_path,batch,computation, metric, freqmod)
        
        
            #layers = ['fc1','fc2','flatten']
            for layer in layers:
        
            
                print('##### current layer is: ', layer)
                #une fonction qui pour la couche et seulement la couche, stocke les activations de toutes les images
                #elle retourne l'array des activations à la couche choisie
                dict_activations = {}
                get_activation_by_layer(activations,batch,dict_activations,computation, metric, k, layer)
            
                df_metrics = pandas.DataFrame.from_dict(dict_activations)
                for index, row in df_metrics.iterrows():
        
                    df = pandas.DataFrame.from_dict(dict(zip(row.index, row.values))).T
        
        
                    os.makedirs(path+"", exist_ok=True)
                    #l'enregistrer dans results, en précisant la layer dans le nom
                    if count == 0:
                        df.to_csv(path+"/"+namefile+"_values_"+layer+".csv")
                    else:
                        df.to_csv(path+"/"+namefile+"_values_"+layer+".csv",mode='a', header=False)



    t0 = time.time()

    allCFD = False
    if bdd == "CFD_ALL":
        allCFD = True
        bdd = "CFD"


    if computer == 'LINUX-ES03':
        computer = '../../'


    labels_path, images_path, log_path = getPaths(bdd, computer)
    model, layers, flatten_layers =configModel(model_name, weight)
    
    #sur le mesoLR, le chemin d'écriture et de lecture est différent
    if computer == '/home/tieos/work_cefe_swp-smp/melvin/thesis/': 
            computer = '/lustre/tieos/work_cefe_swp-smp/melvin/thesis/'
    
   

        # adapte le chemin suivant la methode 
    # 
    if metric == 'mean':
        path= computer+"results"+"/"+bdd+"/average"
        namefile =  "average"
    elif metric == 'max':
        path= computer+"results"+"/"+bdd+"/max"
        namefile =  "max"
    print("path :", computer)
    #if computation == 'featureMap' and metric = 'acp':
    #    path= path+"_FeatureMap"
    
    #dict_compute_pc = {}   #un dictionnaire qui par couche, a ses composantes principales (et les coorodnnées de chaque image pour chaque composante)
    #dict_labels = {}
       ##########

    if allCFD == True:
        combinaison = getAllGenreEthnieCFD(labels_path, exception= {'ethnie' : ["M","I"]})

        for key in combinaison.keys():
            if key == "":
                bdd = "CFD"
            else:
                bdd = "CFD_"+key
            imglist = combinaison[key]
            #adapte le chemin suivant la methode 
            if metric == 'mean':
                path= computer+"results"+"/"+bdd+"/average"
                namefile =  "average"
            elif metric == 'max':
                path= computer+"results"+"/"+bdd+"/max"
                namefile =  "max"
            print("path :", computer)
            average()
    else:
        if bdd == "Fairface":
            #filt = {'ethnie' : "Asian", 'genre' : "Female"}
            filt = {'ethnie' : "White", 'genre' : "Male"}
            imglist = parserFairface(labels_path,filt)
            #imglist = parserFairface(labels_path)
            for key, item in filt.items():
                if bdd == "Fairface":
                    bdd = bdd+"_"
                bdd = bdd+item[0]
        else:
            imglist = [f for f in os.listdir(images_path)]
        print("longueur imglist: ", len(imglist))

        average()


    
    

        
    if '/lustre/tieos/work_cefe_swp-smp/melvin/thesis/' in path:
        path = '/home/tieos/work_cefe_swp-smp/melvin/thesis/'

    
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
    """! parcours tout les fichier du repertoire path dans l'ordre indiquer dans formatOrdre
    @param path chemin du répertoire
    @param formatOrdre [optionnel] permet de parcourir le repertoire dans un ordre précis:
        syntaxe: formatOrdre[  prefixe, TabName[], sufixe]

    @return liste des nom des fichiers
    """
    files = []
    if len(formatOrdre)==0: #ordre de parcours alphabétique
        files = [f for f in os.listdir(path)]    
    else: #parcours les fichier qui match avec formatOrdre
        #if isinstance(formatOrdre[0], list):
            files = [formatOrdre[0]+f+formatOrdre[2] for f in formatOrdre[1]]
    return files


def eachFilePlot(path, formatOrdre = []):
    """! affiche un histogramme des fichier parcouru 
    [obsolete]
    @param path chemin du répertoire
    @param formatOrdre [optionnel] permet de parcourir le repertoire dans un ordre précis:
        syntaxe: formatOrdre[  prefixe, TabName[], sufixe]

    @return liste des nom des fichiers
    """
    files =  getAllFile(path, formatOrdre)
    for each in files:
        csv_path = path + "/" + each
        if os.path.isfile(csv_path):
            
            x, head = readCsv(csv_path) #recupère le CSV
            
            print('######', each,"     ")
            plots.plotHist_fromFiles(x, head, name =each )

def eachFile(path, formatOrdre = []):
    #[obsolete]
    files =  getAllFile(path, formatOrdre)
    tabcsv =[]
    for each in files:
        csv_path = path + "/" + each
        if os.path.isfile(csv_path):
            
            x, head = readCsv(csv_path) #recupère le CSV
            tabcsv.append(x)
    return tabcsv


def eachFileCSV(path, formatOrdre = [],writeLLH = False, pathModel = "", method = "pca", pathLLH = ""):
    """! parcours tout les chifier du repertoire path, fait: mixureGaussian, LLH, tableau des nbe de PC par couche
    
    @param path chemin des CSV a traiter
    @param formatOrdre permet de parcourir dans un ordre précis:
        syntaxe: formatOrdre[  prefixe, TabName[], sufixe]
    @param pathForLLH path pour les log likeliHood

    @return tableau des nbe de PC par couche
    """
    tabPC = []
    pathPCA = path+"/"+method
  #  pathPCA = path+"/"+"pca_FeatureMap"
    #pathModel = path+"/"+"average_FeatureMap"
    

    pathHist = path+"/"+"histo"
        #sur le mesoLR, le chemin d'écriture et de lecture est différent
    if pathLLH == "": 
        pathLLH = path

    files = getAllFile(pathPCA, formatOrdre)
    if pathModel !="":
        pathLLH = pathLLH+"/"+"LLH_"+method+"_model"
        filesModel = getAllFile(pathModel, formatOrdre)
    else:
        pathLLH = pathLLH+"/"+"LLH_"+method
     
    bgm = True
    if bgm == True:
        pathLLH=pathLLH+"_bgm"

    #pathLLH= pathLLH+"_bgm"
    #label, _ = readCsv(pathLabel, True)
    #label = np.transpose(label)[0]
    arrayIntra = []
    arrayInter = []
    #files =    ["average_values_fc1.csv","average_values_fc2.csv", "average_values_flatten.csv"]
    for each in files:
    #for each in ["average_values_fc1.csv","average_values_fc2.csv","average_values_flatten.csv"]:
        csv_path = pathPCA + "/" + each
        x, _ = readCsv(csv_path) #recupère le CSV
        tabPC.append(x.shape[1])
        if pathModel !="":
            model, _ = readCsv(pathModel+ "/" + each)#, intervalle = [0,700])
        else:
            model = x #getSousBDD(x, label)

        #lll = model.shape[0]//2
        #print('######', each,"     ", x.shape[1])
       
        if bgm == True:
           
            gm =metrics_melvin.getBayesianGaussian(model, nbMaxComp = 15)
        else:
            gm = metrics_melvin.getMultigaussian(model,name =  pathPCA+" "+each, plot = False, nbMaxComp = 15) #min(12,model.shape[0]//2))
        print("gauss")
        #metrics.doVarianceOfGMM(gm, x)
        allLLH =  metrics_melvin.DoMultipleLLH(gm, model,101,x)
#       metrics_melvin.doVarianceOfGMM(allLLH, plot = True)
        
        allVar = np.var(allLLH, axis=0) # récup la variance intraImage
        varExtra = np.var(allLLH, axis=1) # variance interImage
        intraMoy =  np.mean(allVar)
        interMoy = np.mean(varExtra)
        arrayIntra.append(intraMoy)
        arrayInter.append(interMoy)

        print(each," ;  intraMoyenne = ",intraMoy,"    ;  interMoyenne = ",interMoy)
        
        #allLLH2 =  metrics_melvin.DoMultipleLLH(gm, x,100)
        #CompareAndDoMedian(allLLH,allLLH2)
        

        #metrics_melvin.doHist(allLLH, plot = True, name = "distributions des LLH pour GMM")
        #plots.plot_correlation(allLLH, name = "correlation entre BGM", nameXaxis="BGM",nameYaxis="BGM")
        allLLH = np.array([np.median(allLLH, axis=0)])
        #plots.plotHist(np.array([LLH_KDE,LLH_GMM[0]]), name= "distribution des LLH\n KDE")
        #allLLH = metrics_melvin.chooseBestComposante(allLLH)
        #allLLH =metrics.removeOutliers(allLLH)
        #allHist, legend = metrics_melvin.FdoHist(allLLH, false, "distributions des LLH pour GMM")
        #metrics.writeHist(allHist, legend,pathHist,"_nbComp="+str(gm.n_components)+"_covarType="+gm.covariance_type+"_"+each)
            #sur le mesoLR, le chemin d'écriture et de lecture est différent

        if writeLLH:
            import re
            #regex = re.search("((?:pca_values){1})(.*\.csv$)",each)
            regex = re.search("((?:_values){1})(.*\.csv$)",each)
            layer = regex.group(2)

            metrics_melvin.writeLikelihood(allLLH, pathLLH, layer)
       
    
    #plots.plotPC([arrayIntra, arrayInter], ["intra", "inter"], files, title = "moyenne des variance intra et inter image par couche");        
    return tabPC



#Melvin [Obsolete]
def eachFileCSV_Kernel(path, filesPC):
    """! effectue l'opération KDE pour chaque couche de la bdd indiqué dans path, des fichier filesPC
    
    """
    #tabPC = []
    pathPCA = path+"/"+"pca"
    
    #pathHist = path+"/"+"histo"

    #files = getAllFile(pathPCA, formatOrdre)
   
    for each in filesPC:
        x, _ = readCsv(pathPCA + "/" + each)  #recupère le CSV
        kde= metrics_melvin.KDE(x)

        AllLLH =  metrics_melvin.DoMultipleLLH(kde, x,1, x)

        metrics_melvin.doHist(AllLLH, plot = True, name = "distributions des LLH pour KDE")



#Melvin [Obsolete]
def each_compare_GMM_KDE(path, filesPC):
    """! KDE abandonné, obsolete
    Fonction mère effectuant toute la pipeline de test entre la méthode GMM et KDE
    """

    pathPCA = path+"/"+"pca"
    
    AllSpearman = []
    AllPearson = []
    for each in filesPC:
        
        x, _ = readCsv(pathPCA + "/" + each)  #recupère le CSV
        if x is None:
            continue
        x = metrics_melvin.centreReduit(x)
        kde = metrics_melvin.KDE(x, False)
        LLH_KDE =  metrics_melvin.DoMultipleLLH(kde, x,100, x)[0]
        #LLH_2 = np.median(LLH_KDE, axis=0)
        gm = metrics_melvin.getMultigaussian(x,name =  pathPCA+" "+each, plot=[False,False], nbMaxComp =10)
        
       # metrics.doVarianceOfGMM(gm, x)
        LLH_GMM =  metrics_melvin.DoMultipleLLH(gm, x,100, x)
        LLH_GMM = np.median(LLH_GMM, axis=0)
       # LLH_GMM =metrics.removeOutliers(LLH_GMM)
        
        metrics_melvin.doHist([LLH_GMM,LLH_KDE], plot = True, name = "histogramme GMM et KDE")

        #plots.plot_correlation([LLH_GMM,LLH_KDE], name = "correlation GMM et KDE", nameXaxis="GMM",nameYaxis="KDE")
        AllSpearman.append( metrics_melvin.spearman( LLH_GMM,LLH_KDE))
        AllPearson.append( metrics_melvin.pearson( LLH_GMM,LLH_KDE))
       # plots.plotHist(np.array([LLH_KDE,LLH_GMM[0]]), name= "distribution des LLH\n KDE         |           GMM", max = 2)
       # metrics.compareValue(LLH_KDE, LLH_GMM[0], "Difference LLH entre GMM et KDE")
        #metrics.CompareOrdre(LLH_KDE, LLH_GMM[0], "Difference d'ordre LLH entre GMM et KDE")
        #metrics.doHist(np.array([LLH]), plot = True)
    print( AllSpearman)
    writeCSV=False
    if writeCSV:
        pathSpearman = path+"/"+"spearman"
        df = pandas.DataFrame(AllSpearman)
        df = df.transpose()
        df.columns = filesPC
        df = df.transpose()
        os.makedirs(pathSpearman, exist_ok=True)
        df.to_csv(pathSpearman+"/corr_spearman.csv")
    return AllSpearman, AllPearson





#Melvin
def readCsv(path,noHeader = False,noNumerate = True, intervalle = []):
    """! Lit un fichier .csv, convertit toutes les valeurs en float et retourne un numpyArray
    @param path         chemin du csv
    @param noHeader [optionnel]     boolean, ignore la premiere ligne si True
    @param noNumerate [optionnel]  boolean, ignore la premiere colonne si True
    @param interavelle [optionnel] tableau de 2 nombres, extrait les donnée du csv entre les ligne intervalle[0] et intervalle[1]
    @return tableau numpy, list       retourne le tableau des donnée, et la liste de header du csv 
    """
    try:
        with open(path, newline='') as csvfile:
            rows = list(csv.reader(csvfile,delimiter=','))
            #rows[0].pop(0)

            if  intervalle:
                rows = rows[intervalle[0]:intervalle[1]]
           
            for row in rows[0:]:

                for i in range(len(row)):
                    try:
                        row[i] = float(row[i])
                    except:
                        pass
                if noNumerate:
                    row.pop(0)
            if not noHeader:
                head = rows.pop(0)
            else:
                head = rows[0]
            return np.array(rows), head
    except OSError:
        print("cannot open", path)
        return None, None
    else:
        print("an error has occurred ")
        return None, None


#Melvin
def CompareAndDoMedian(ArrayA,ArrayB):
    """! Compare la moyenne de deux tableaux et affiche la correlation

    """
    mA = np.median(ArrayA,axis=0)
    mB = np.median(ArrayB,axis=0)
    plots.plot_correlation([mA,mB])


#obsolete, verifier avant d'enlever
#def getSousBDD(acp, label, min = 0, max=100):
#    if min==0 and max ==100:
#        min = np.quantile(label, .50)
#    filterLabel = np.where((label >min) & (label < max), True, False)
#    model = acp[filterLabel]
#    return model



#Melvin
def parserFairface(path, filt = {'genre' : "Female", 'ethnie' : "Asian"}):
    """! filtre Fairface par rapport a l'ethnie et au genre
    @param path     chemin de fairface
    @param filt     dictionnaire, extrait les images par etnie et par genre indiqué dans le dictionnaire
    @return         retourne la liste des images filtrer par ethnie et genre
    """
    x, head = readCsv(path,noHeader = False,noNumerate = False)  #recupère le CSV
    filtered = np.array(list(filter(lambda val:(
       (filt.get('genre','') in val or len(filt.get('genre','')) == 0) #soit match soit list vide
       and 
       (filt.get('ethnie','') in val[3] or len(filt.get('ethnie','')) == 0)
       and
       ( val[1] in ["more than 70","10 - 19", "3 - 9"])  #à tester
       )
                                    , x)))
    #filtered = np.array(list(filter(lambda val:( filt[1] in val[3] ), x)))
#    filtered = np.array(list(filter(lambda val:True, x)))
    # and print(filt[1]," et ", val)
    print(filtered[-1])
    return filtered[:,0]



#Melvin
def getAllGenreEthnieCFD(path, exception= {'ethnie' : ["M","I"]}):
    """! Extrait chaque éthnie et genre de CFD et retourne un dictionnaire {sous ensemble, liste images}
    @param path     chemin de CFD
    @param [facultatif] exception       Dictionnaire, ignore les ethnie du dictionnaire
    @return         retourne un dictionnaire {sous ensemble : list images}
    """

    x, head = readCsv(path,noHeader = False,noNumerate = False)  #recupère le CSV
    CategoryCFD = {"ethnie" : [""], "genre" : [""]}
    #CategoryCFD = {}
    for row in x:
        if not row[0][4] in CategoryCFD["ethnie"] and not row[0][4] in exception["ethnie"]:
            CategoryCFD["ethnie"].append(row[0][4])
        if not  row[0][5] in CategoryCFD["genre"]:
            CategoryCFD["genre"].append(row[0][5])
   
    combinaison = {}

    for i in reversed(CategoryCFD["ethnie"]):
        for j in reversed(CategoryCFD["genre"]):
            #combinaison.append(i+j)
            combinaison[i+j] = parserCFD(path, filt = {'genre' : j, 'ethnie' : i})
            #parserCFD(path, filt = {'genre' : j, 'ethnie' : i})
    print("ma")
    return combinaison
    


#Melvin
def parserCFD(path, filt = {'genre' : "F", 'ethnie' : "A"}):
    """! filtre CFD par rapport a l'ethnie et au genre
    @param path     chemin de CFD
    @param filt     dictionnaire, extrait les images par etnie et par genre indiqué dans le dictionnaire
    @return         retourne la liste des images filtrer par ethnie et genre
    """
    x, head = readCsv(path,noHeader = False,noNumerate = False)  #recupère le CSV
    #si filtre pas empty:
    filtered = np.array(list(filter(lambda val:(
        (filt.get('genre','') in val[0][5] or len(filt.get('genre','')) == 0) #soit match soit list vide
        and 
        (filt.get('ethnie','') in val[0][4] or len(filt.get('ethnie','')) == 0)
        )
                                    , x)))
    #filtered = np.array(list(filter(lambda val:( filt[1] in val[3] ), x)))
#    filtered = np.array(list(filter(lambda val:True, x)))
    # and print(filt[1]," et ", val)
    print(filtered[-1])

    writeLabelCFD(path, filtered,filt["ethnie"]+filt["genre"])

    return filtered[:,0]

def writeLabelCFD(path, label_Filtered,name):
    #'../../data/redesigned/CFD/labels_CFD.csv'
    df = pandas.DataFrame(label_Filtered,)
    

    dirPath = path.rsplit("/", 1)[0]+"_ALL"       
    os.makedirs(dirPath, exist_ok=True)
        #l'enregistrer dans results, en précisant la layer dans le nom
    df.to_csv(dirPath+"/"+"labels_CFD_"+name+".csv",header=False, index= False)