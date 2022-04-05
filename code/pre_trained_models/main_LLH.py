
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
pathData = '../../'
if len(sys.argv) >1:
    if sys.argv[1]== 'mesoLR':
        sys.path.insert(1,'/home/tieos/work_swp-gpu/melvin/thesis/code/functions')
        pathData = '/home/tieos/work_swp-gpu/melvin/thesis/'
    elif sys.argv[1] == 'sonia':
        pathData =  '/media/sonia/DATA/data_nico/'

import sparsenesslib.high_level as hl
#import sparsenesslib.metrics as metrics
import sparsenesslib.metrics_melvin as metrics_melvin
import sparsenesslib.plots as plots
import pandas as pd
import numpy as np
#####################################################################################
#SETTINGS:
#####################################################################################
PIL.Image.MAX_IMAGE_PIXELS = 30001515195151997
478940                             
#'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
list_bdd = ['MART'] #"['CFD','MART','JEN','SCUT-FBP','SMALLTEST','BIGTEST']"
#list_bdd =['CFD','MART','JEN','SCUT-FBP']
#list_bdd =['BIGTEST']

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

            path = "../../results"+"/"+bdd;
            pathLLH = path+"/"+"LLH"
            _, layers, _ = hl.configModel(model_name, weight)
            #hl.eachFileCSV(path,["pca_values_",layers,".csv"], [pathData,bdd,'_'])
            
            filesLLH = hl.getAllFile(path+"/"+"pca", ["LLH__",layers,".csv"])
            
            alldf = pd.DataFrame()
            
            for each, name in zip(filesLLH,layers):
                csv_path = pathLLH + "/" + each
                x, _ = hl.readCsv(csv_path)
                #df = pd.DataFrame(x)
                a = np.transpose(x)[0]
                alldf[name] = np.transpose(x)[0]
            alldf = alldf.transpose()
            metrics_melvin.writeLikelihood(alldf, pathLLH, bdd+"_AllLLH.csv")