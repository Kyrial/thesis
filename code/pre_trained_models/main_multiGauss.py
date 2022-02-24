#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#[EN] fichier dans le cadre du stage de Melvin
#[FR]

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
import sparsenesslib.high_level as hl
import sparsenesslib.metrics as metrics
#####################################################################################
#SETTINGS:
#####################################################################################
PIL.Image.MAX_IMAGE_PIXELS = 30001515195151997
478940                             
#'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
list_bdd = ['CFD'] #"['CFD','MART','JEN','SCUT-FBP','SMALLTEST','BIGTEST']"
model_name = 'VGG16'  # 'vgg16, resnet (...)'
#weights = 'vggface' #'imagenet','vggface'
list_weights = ['imagenet'] #['vggface','imagenet','vggplace']
computer = 'LINUX-ES03' #no need to change that unless it's sonia's pc, that infamous thing; in which case, put 'sonia' in parameter.
freqmod = 100 #frequency of prints, if 5: print for 1/5 images



#####################################################################################
#CODE
#####################################################################################
list_metrics = ['acp']
k = 1
l = len(list_bdd)*len(list_weights)*len(list_metrics)
for bdd in list_bdd:
    for weight in list_weights:
        for metric in list_metrics:
            print('###########################--COMPUTATION--#################################_STEP: ',k,'/',l,'  ',bdd,', ',weight,', ',metric)
            #path = "../../results"+"/"+bdd+"/"+"pcaBlock"+"/"+"pca_values_"+"block1"+".csv";
           # path = "../../results"+"/"+bdd+"/"+"pca"+"/"+"pca_values_"+"block1_conv1"+".csv";
            path = "../../results"+"/"+bdd+"/"+"pca";
            #x = metrics.readCsv(path)
           # metrics.getMultigaussian(x, bdd+" "+"pcaBlock"+" "+"block1")
            #metrics.getMultigaussian(x, bdd+" "+"pcaBlock"+" "+"block1_conv1")
            hl.eachFileCSV(path)
                                     
            k += 1
#####################################################################################
