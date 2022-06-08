
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
from sklearn import preprocessing
import sparsenesslib.metrics_melvin as metrics_melvin
import sparsenesslib.plots as plots
#####################################################################################
#SETTINGS:
#####################################################################################
PIL.Image.MAX_IMAGE_PIXELS = 30001515195151997
478940                             
#'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
list_bdd = ['MART'] #"['CFD','MART','JEN','SCUT-FBP','SMALLTEST','BIGTEST']"
#list_bdd = ['CFD_AF','CFD_F']
#list_bdd =['CFD','MART','JEN','SCUT-FBP']
#list_bdd =['SCUT-FBP']

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


#method = "_FeatureMap"
method = "_average"
#method = "_max"
method = "_pca"








def do_correlation_LLH(LLH1, LLH2, hist =True):
   # metrics_melvin.doHist([LLH1,LLH2], plot = True, name = "histogramme GMM et KDE")
 
    plots.plot_correlation([LLH1,LLH2], name = "correlation LLH global et LLH/featureMap ", nameXaxis="LLH global",nameYaxis="LLH/featureMap")



def each2LLH(path1, path2):
    layers = [#'input_1',
        'block1_conv1','block1_conv2','block1_pool','block2_conv1', 'block2_conv2','block2_pool',
            'block3_conv1','block3_conv2','block3_conv3','block3_pool','block4_conv1','block4_conv2','block4_conv3',
            'block4_pool', 'block5_conv1','block5_conv2','block5_conv3','block5_pool']
    AllSpearman = []
    AllPearson = []
    filesLLH = hl.getAllFile("", ["LLH__",layers,".csv"])
    for each, name in zip(filesLLH,layers):
            LLH1, _ =hl.readCsv(path1 + "/" + each)
            LLH2, _= hl.readCsv(path2 + "/" + each)
            LLH1 =np.transpose(LLH1)[0]
            LLH2 =np.transpose(LLH2)[0]
           # do_correlation_LLH(LLH1, LLH2)
            s, _ = metrics_melvin.spearman( LLH1, LLH2)
            p, _ = metrics_melvin.pearson( LLH1, LLH2)
            AllSpearman.append(s )
            AllPearson.append(p )
      
    plots.plotPC([np.array(AllSpearman),np.array(AllPearson)], ["Spearman","Pearson"], layers, "Correlation pour CFD_AF entre max et Average")




for bdd in list_bdd:
    for weight in list_weights:
        for metric in list_metrics:
            _, layers, _ = hl.configModel(model_name, weight)
            filesLLH = hl.getAllFile("", ["LLH__",layers,".csv"])


            print('###########################--COMPUTATION--#################################_STEP: ',k,'/',l,'  ',bdd,', ',weight,', ',metric)
            path1 = "../../results"+"/"+bdd+"/"+"LLH"+"_max"
            path2 = "../../results"+"/"+bdd+"/"+"LLH"+method
            #each2LLH(path1, path2)

            path = pathData+"results"+"/"+bdd;
            #pathLLH = path+"/"+"LLH_bestRepetition"
            pathLLH = path+"/"+"LLH"+method# +"_model"
            #_, layers, _ = hl.configModel(model_name, weight)
            #hl.eachFileCSV(path,["pca_values_",layers,".csv"], [pathData,bdd,'_'])
            
            #filesLLH = hl.getAllFile("", ["LLH__",layers,".csv"])
            
            alldf = pd.DataFrame()
            
            for each, name in zip(filesLLH,layers):
                csv_path = pathLLH + "/" + each
                try:
                    x, _ = hl.readCsv(csv_path)
                    #df = pd.DataFrame(x)
                    a = np.transpose(x)[0]
                except Exception :
                    continue
                else:
                    alldf[name] = np.transpose(x)[0]

            #std_scale = preprocessing.StandardScaler().fit(alldf) #centrer reduit
            #LLH_tr2 = std_scale.transform(alldf)
            alldf = alldf.transpose()
            std_scale = preprocessing.StandardScaler().fit(alldf) #centrer reduit
            #LLH_tr = std_scale.transform(alldf)
            #alldf2 = pd.DataFrame(LLH_tr, index = layers)
            metrics_melvin.writeLikelihood(alldf, pathLLH, bdd+"_AllLLH.csv")