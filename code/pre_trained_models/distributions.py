#!/usr/bin/env python
#####################################################################################
# LIBRAIRIES:
#####################################################################################
import time
t0 = time.time()
from tensorflow.keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
import scipy
from scipy import stats
import PIL
import sys
sys.path.insert(1,'../../code/functions')
from sparsenesslib import *
import time
t0 = time.time()
import os
from tensorflow.keras.preprocessing.image import load_img
import keract  # low to import
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import cupy as cp
from numba import jit, cuda
#####################################################################################
#SETTINGS:
#####################################################################################
PIL.Image.MAX_IMAGE_PIXELS = 30001515195151997 #useful for JEN bc images are heavy 
478940                             
bdd = 'CFD' #'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
model_name = 'VGG16'  # 'vgg16, resnet (...)'
weights = 'imagenet' #'imagenet','vggface'
computer = 'sonia'

#####################################################################################
#FUNCTIONS
#####################################################################################

def plot_distributions_activations_layers(model,path,layers):
    """
    save a hist for activations values of each layers
    """
    imgs = [f for f in os.listdir(path)]    

    activations_dict = {}

    for each in imgs:       
          
        img_path = path + "/" + each
        img = load_img(img_path, target_size=(224, 224))
        image = img_to_array(img)
        img = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(img)
        # récupération des activations
        activations = keract.get_activations(model, image)
             
        
        for layer in layers:             
                       
            array = activations[layer].flatten()                                

            if layer in activations_dict:                  
                activations_dict[layer].append(array) 
            else:                 
                activations_dict[layer] = []
                activations_dict[layer].append(array)    
    
    
    for each in activations_dict: 
        print(each)
        fusion = (np.vstack(activations_dict[each])).flatten()         
        kurt = scipy.stats.kurtosis(fusion)         
        title = 'layer: ' + str(each) +  ", kurtosis: " + str(kurt)         
        plt.hist(fusion, bins = 30)        
        plt.title(title, fontsize=10)                 
        plt.savefig(log_path +'_' + each + '_graphe')
        plt.clf()
####################################################################################
plot_distributions_activations_layers(model,images_path,layers)






"""
    for each in activations_dict: 
        fusion = np.vstack(activations_dict[each])
        fusion = fusion.flatten()
        kurt = scipy.stats.kurtosis(fusion)    
        title = 'layer: ' + str(each) +  ", kurtosis: " + str(kurt)   
        axs[j, i].hist(fusion, bins = 30)
        if (i+1)%3 == 0:
            i = 0
            j += 1
        else:
            i += 1         
        plt.title(title, fontsize=10) 
                  
        #plt.savefig(log_path +'_' + layer + 'graphe')
    plt.show()
"""