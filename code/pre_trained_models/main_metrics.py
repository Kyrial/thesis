#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#[EN]Main program for calculating sparsity correlated metrics on neural network middle layers. 
#Loop on the combinatorics of the parameters (databases, weights, model etc)
#Choice of these parameters below. 

#[FR]Programme principal du calcul de métriques corrélées à la sparsité sur les couches intermédiaires de réseaux de neurones. 
# Boucle sur les combinatoires des paramètres (bases de données, poids, modèle etc)
# Choix de ces paramètres ci dessous. 

#####################################################################################
# LIBRAIRIES:
#####################################################################################
#public librairies
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import PIL
import sys
#personnal librairies
sys.path.insert(1,'../../code/functions')

cwd = os.getcwd()
import sparsenesslib.high_level as hl
#####################################################################################
#SETTINGS:
#####################################################################################
PIL.Image.MAX_IMAGE_PIXELS = 30001515195151997
aaa = 5+ 7+99
478940                             
#'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
list_bdd = ['BIGTEST'] #"['CFD','MART','JEN','SCUT-FBP','SMALLTEST','BIGTEST']"
model_name = 'VGG16'  # 'vgg16, resnet (...)'
#weights = 'vggface' #'imagenet','vggface'
list_weights = ['imagenet'] #['vggface','imagenet','vggplace']
list_metrics = ['L0'] #['L0','L1','gini_flatten','gini_channel','gini_filter','kurtosis']
computer = 'LINUX-ES03' #no need to change that unless it's sonia's pc, that infamous thing; in which case, put 'sonia' in parameter.
freqmod = 10 #frequency of prints, if 5: print for 1/5 images
#####################################################################################
#CODE
#####################################################################################
k = 1
l = len(list_bdd)*len(list_weights)*len(list_metrics)
for bdd in list_bdd:    
    for weight in list_weights:
        for metric in list_metrics:
            print('###########################--COMPUTATION--#################################_STEP: ',k,'/',l,'  ',bdd,', ',weight,', ',metric)
            hl.extract_metrics(bdd,weight,metric, model_name, computer, freqmod,k)            
            k += 1
#####################################################################################
