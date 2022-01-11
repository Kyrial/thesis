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
#import vggplaces.vgg16_places_365 as places
#personnal librairies
sys.path.insert(1,'../../code/functions')
import sparsenesslib.keract as keract
import sparsenesslib.metrics as metrics
import sparsenesslib.sparsenessmod as spm
import sparsenesslib.activations_structures as acst
#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################
def compute_sparseness_metrics_activations(model, flatten_layers, path, dict_output, layers, computation, formula, freqmod,k):
    '''
    compute metrics of the layers given in the list *layers*
    of the images contained in the directory *path*
    by one of those 3 modes: flatten channel or filter (cf activations_structures subpackage)
    and store them in the dictionary *dict_output*.
    '''
    imgs = [f for f in os.listdir(path)]    
    i = 1
    for each in imgs:

        if i%freqmod == 0:
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)
        i += 1
        img_path = path + "/" + each
        img = load_img(img_path, target_size=(224, 224))
        image = img_to_array(img)
        img = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))  
        image = preprocess_input(img)
        # récupération des activations
        activations = keract.get_activations(model, image)
        activations_dict = {}
        acst.compute_activations(layers, flatten_layers, computation, activations, activations_dict,formula,k)
        dict_output[each] = activations_dict
#####################################################################################
def parse_activations_by_layer(model,path, dict_output, layer, computation, formula, freqmod,k):
    
    '''
    une fonction qui pour la couche et seulement la couche, stocke les activations de toutes les images
    elle retourne l'array des activations à la couche choisie    
    '''
    print("1")
    imgs = [f for f in os.listdir(path)] 
    print("2")   
    i = 1
    print("3")

    for each in imgs:
        print("4")

        if i%freqmod == 0:
            print("5")
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)
        i += 1
        print("6")
        img_path = path + "/" + each
        print("7")
        img = load_img(img_path, target_size=(224, 224))
        print("8")
        image = img_to_array(img)
        print("9")
        img = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))  
        print("10")
        image = preprocess_input(img)
        print("11")
        # récupération des activations
        activations = keract.get_activations(model, image)
        print("12")
        activations_dict = {}
        print("13")
        acst.compute_flatten(activations, activations_dict, layer, formula,k)  
        print("14")
        dict_output[each] = activations_dict
        print("15")
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

    df_metrics = df_metrics.rename(columns = {'input_2': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_3': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_4': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_5': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_6': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_7': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_8': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_9': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_10': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_11': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_12': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_13': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_14': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_15': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_16': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_17': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_18': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_19': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_20': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_21': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_22': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_23': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_24': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_25': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_26': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_27': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_28': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_29': 'input_1'})
    df_metrics = df_metrics.rename(columns = {'input_30': 'input_1'})

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

    if computer == 'sonia': #databases aren't in repo bc they need to be in DATA partition of the pc (more space)
        if bdd == 'CFD':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/CFD/labels_CFD.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/CFD/images'
            log_path ='../../results/CFD/log_'
        elif bdd == 'JEN':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/JEN/labels_JEN.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/JEN/images'
            log_path ='../../results/JEN/log_'
        elif bdd == 'SCUT-FBP':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/SCUT-FBP/images'
            log_path ='../../results/SCUT-FBP/log_'
        elif bdd == 'MART':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/MART/labels_MART.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/MART/images'
            log_path ='../../results/MART/log_'
        elif bdd == 'SMALLTEST':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/small_test/labels_test.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/small_test/images'
            log_path ='../../results/smalltest/log_'
        elif bdd == 'BIGTEST':        
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/big_test/labels_bigtest.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/big_test/images'
            log_path ='../../results/bigtest/log_'

    else: #for others configurations,all paths are relative paths in git repository
        if bdd == 'CFD':
            labels_path ='../../data/redesigned/CFD/labels_CFD.csv'
            images_path ='../../data/redesigned/CFD/images'
            log_path ='../../results/CFD/log_'
        elif bdd == 'JEN':
            labels_path ='../../data/redesigned/JEN/labels_JEN.csv'
            images_path ='../../data/redesigned/JEN/images'
            log_path ='../../results/JEN/log_'
        elif bdd == 'SCUT-FBP':
            labels_path ='../../data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
            images_path ='../../data/redesigned/SCUT-FBP/images'
            log_path ='../../results/SCUT-FBP/log_'
        elif bdd == 'MART':
            labels_path ='../../data/redesigned/MART/labels_MART.csv'
            images_path ='../../data/redesigned/MART/images'
            log_path ='../../results/MART/log_'
        elif bdd == 'SMALLTEST':                       
            labels_path ='../../data/redesigned/small_test/labels_test.csv'
            images_path ='../../data/redesigned/small_test/images'
            log_path ='../../results/smalltest/log_'            
        elif bdd == 'BIGTEST':
            labels_path ='../../data/redesigned/big_test/labels_bigtest.csv'
            images_path ='../../data/redesigned/big_test/images'
            log_path ='../../results/bigtest/log_'  

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

    t0 = time.time()

    if computer == 'sonia': #databases aren't in repo bc they need to be in DATA partition of the pc (more space)
        if bdd == 'CFD':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/CFD/labels_CFD.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/CFD/images'
            log_path ='../../results/CFD/log_'
        elif bdd == 'JEN':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/JEN/labels_JEN.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/JEN/images'
            log_path ='../../results/JEN/log_'
        elif bdd == 'SCUT-FBP':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/SCUT-FBP/images'
            log_path ='../../results/SCUT-FBP/log_'
        elif bdd == 'MART':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/MART/labels_MART.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/MART/images'
            log_path ='../../results/MART/log_'
        elif bdd == 'SMALLTEST':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/small_test/labels_test.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/small_test/images'
            log_path ='../../results/smalltest/log_'
        elif bdd == 'BIGTEST':        
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/big_test/labels_bigtest.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/big_test/images'
            log_path ='../../results/bigtest/log_'

    else: #for others configurations,all paths are relative paths in git repository
        if bdd == 'CFD':
            labels_path ='../../data/redesigned/CFD/labels_CFD.csv'
            images_path ='../../data/redesigned/CFD/images'
            log_path ='../../results/CFD/log_'
        elif bdd == 'JEN':
            labels_path ='../../data/redesigned/JEN/labels_JEN.csv'
            images_path ='../../data/redesigned/JEN/images'
            log_path ='../../results/JEN/log_'
        elif bdd == 'SCUT-FBP':
            labels_path ='../../data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
            images_path ='../../data/redesigned/SCUT-FBP/images'
            log_path ='../../results/SCUT-FBP/log_'
        elif bdd == 'MART':
            labels_path ='../../data/redesigned/MART/labels_MART.csv'
            images_path ='../../data/redesigned/MART/images'
            log_path ='../../results/MART/log_'
        elif bdd == 'SMALLTEST':                       
            labels_path ='../../data/redesigned/small_test/labels_test.csv'
            images_path ='../../data/redesigned/small_test/images'
            log_path ='../../results/smalltest/log_'            
        elif bdd == 'BIGTEST':
            labels_path ='../../data/redesigned/big_test/labels_bigtest.csv'
            images_path ='../../data/redesigned/big_test/images'
            log_path ='../../results/bigtest/log_'  

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

    dict_compute_pc = {}   #un dictionnaire qui par couche, a ses composantes principales (et les coorodnnées de chaque image pour chaque composante)
    dict_labels = {}

    for layer in layers:   

        print('A)')
        print('##### current layer is: ', layer)
        #une fonction qui pour la couche et seulement la couche, stocke les activations de toutes les images
        #elle retourne l'array des activations à la couche choisie
        dict_activations = {}
        print('B')
        
        parse_activations_by_layer(model,images_path,dict_activations, layer, 'flatten', metric, freqmod, k)
        print('C')
        pc = []
        #une fonction qui fait une acp la dessus, qui prends en entrée la liste pc vide et l'array des activations,
        #  et retourne la liste remplie
        metrics.acp_layers(dict_activations, pc)
        print('D')
        #A CODER 

        #dict_compute_pc[layer] = pc
        
    
    spm.parse_rates(labels_path, dict_labels)
    
    today = date.today()
    today = str(today)

    

    
#####################################################################################
def analyse_metrics(model_name, computer, bdd, weight, metric,k):
    
    #récupération du nom des couches
    if model_name == 'VGG16':
        if weight == 'imagenet':
            layers = ['input_1','block1_conv1','block1_conv2','block1_pool','block2_conv1', 'block2_conv2','block2_pool',
            'block3_conv1','block3_conv2','block3_conv3','block3_pool','block4_conv1','block4_conv2','block4_conv3',
            'block4_pool', 'block5_conv1','block5_conv2','block5_conv3','block5_pool','flatten','fc1', 'fc2'] 
        elif weight == 'vggface':
            layers = ['input_1','conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2','conv3_3',
            'pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5','flatten',
            'fc6/relu','fc7/relu']
        elif weight == 'vggplaces':            
            layers = ['input_1','block1_conv1','block1_conv2','block1_pool','block2_conv1', 'block2_conv2','block2_pool',
            'block3_conv1','block3_conv2','block3_conv3','block3_pool','block4_conv1','block4_conv2','block4_conv3',
            'block4_pool', 'block5_conv1','block5_conv2','block5_conv3','block5_pool','flatten','fc1', 'fc2']
    elif model_name == 'resnet50':
        if weight == 'imagenet': 
            print('error, model not configured')
        elif weight == 'vggfaces':
            print('error, model not configured')  

    #path d'enregistrement des résultats
    if computer == 'sonia': #databases aren't in repo bc they need to be in DATA partition of the pc (more space)
        if bdd == 'CFD':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/CFD/labels_CFD.csv'
            log_path ='../../results/CFD/log_'
        elif bdd == 'JEN':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/JEN/labels_JEN.csv'
            log_path ='../../results/JEN/log_'
        elif bdd == 'SCUT-FBP':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
            log_path ='../../results/SCUT-FBP/log_'
        elif bdd == 'MART':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/MART/labels_MART.csv'
            log_path ='../../results/MART/log_'
        elif bdd == 'SMALLTEST':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/small_test/labels_test.csv'
            log_path ='../../results/smalltest/log_'
        elif bdd == 'BIGTEST':        
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/big_test/labels_bigtest.csv'
            log_path ='../../results/bigtest/log_'

    else: #for others configurations,all paths are relative paths in git repository
        if bdd == 'CFD':
            labels_path ='../../data/redesigned/CFD/labels_CFD.csv'
            log_path ='../../results/CFD/log_'
        elif bdd == 'JEN':
            labels_path ='../../data/redesigned/JEN/labels_JEN.csv'
            log_path ='../../results/JEN/log_'
        elif bdd == 'SCUT-FBP':
            labels_path ='../../data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
            log_path ='../../results/SCUT-FBP/log_'
        elif bdd == 'MART':
            labels_path ='../../data/redesigned/MART/labels_MART.csv'
            log_path ='../../results/MART/log_'
        elif bdd == 'SMALLTEST':                       
            labels_path ='../../data/redesigned/small_test/labels_test.csv'
            log_path ='../../results/smalltest/log_'            
        elif bdd == 'BIGTEST':
            labels_path ='../../data/redesigned/big_test/labels_bigtest.csv'
            log_path ='../../results/bigtest/log_'  

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

