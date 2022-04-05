#####################################################################################
# 1. DESCRIPTION:
#####################################################################################

#####################################################################################
# 2. LIBRAIRIES:
#####################################################################################
library("rjson")
library("purrr")
library("tidyr")
library("tibble")
library("plyr")
library("corrplot")
library("FactoMineR")
library("dplyr")
setwd("/home/renoult/Bureau/thesis/code/functions")
#####################################################################################
# 3. PARAMETRES: def analyse_metrics(model_name, bdd, weight, metric,k):
#####################################################################################
#####################################################################################
# 3.1 Parametres
#####################################################################################
#mettre ça pas en dur a terme mais en paramètres passé au script python (ou pas?)

model_name <- 'VGG16'
bdd <- 'CFD'
weight <- 'imagenet'
metric <- 'gini_flatten'


        
        #####################################################################################
        # 3.2. Data management
        #####################################################################################
          layers = c('input_1',
                       'block1_conv1','block1_conv2','block1_pool',
                       'block2_conv1','block2_conv2','block2_pool',
                       'block3_conv1','block3_conv2','block3_conv3','block3_pool',
                       'block4_conv1','block4_conv2','block4_conv3','block4_pool',
                       'block5_conv1','block5_conv2','block5_conv3','block5_pool',
                       'flatten','fc1', 'fc2')
          
         
        #path d'enregistrement des résultats et chargement des données  
        if (bdd == 'CFD'){
          labels_path ='../../data/redesigned/CFD/labels_CFD.csv'
          log_path ='../../results/CFD/log_'
          
        }else if (bdd == 'JEN'){
          labels_path ='../../data/redesigned/JEN/labels_JEN.csv'
          log_path ='../../results/JEN/log_'
          
        }else if (bdd == 'SCUT-FBP'){
          labels_path ='../../data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
          log_path ='../../results/SCUT-FBP/log_'
          
        }else if (bdd == 'MART'){
          labels_path ='../../data/redesigned/MART/labels_MART.csv'
          log_path ='../../results/MART/log_'
          
        }else if (bdd == 'SMALLTEST'){                       
          labels_path ='../../data/redesigned/small_test/labels_test.csv'
          log_path ='../../results/smalltest/log_'  
          
        }else if (bdd == 'BIGTEST'){
          labels_path ='../../data/redesigned/big_test/labels_bigtest.csv'
          log_path ='../../results/bigtest/log_'  
        }
        
        matrix_metrics <- do.call(cbind, fromJSON(file = paste(log_path,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
        colnames(matrix_metrics)[2] <- 'input_1'
        
        df_metrics <- as.data.frame(matrix_metrics, optional = TRUE)
        df_metrics = sapply(df_metrics, as.numeric)
        

        df_metrics <- as.data.frame(df_metrics)


        #################
        # 5. MULTIPLES MODELS
        #####################################################################################
        
        #faire des modèles avec les métriques intermédiaires (points d'inflexion), et des interactions comme la complexité par ex)
        if (weight %in% c('imagenet','vggplaces')) {
         df_metrics = rename(df_metrics, c( "input_1" = "input_1" ,
                                          'conv1_1' = 'block1_conv1',
                                          'conv1_2' = 'block1_conv2',
                                         'pool1' =  'block1_pool',
                                         'conv2_1' =  'block2_conv1',
                                          'conv2_2' = 'block2_conv2',
                                         'pool2' =  'block2_pool',
                                          'conv3_1' = 'block3_conv1',
                                          'conv3_2' = 'block3_conv2',
                                         'conv3_3' =  'block3_conv3',
                                         'pool3' =  'block3_pool',
                                          'conv4_1' = 'block4_conv1',
                                          'conv4_2' = 'block4_conv2',
                                          'conv4_3' = 'block4_conv3',
                                          'pool4' = 'block4_pool',
                                          'conv5_1' = 'block5_conv1',
                                          'conv5_2' = 'block5_conv2',
                                         'conv5_3' =  'block5_conv3',
                                          'pool5' = 'block5_pool',
                                          'flatten' = 'flatten',
                                          'fc6/relu' = 'fc1',
                                          'fc7/relu' = 'fc2'))
          
        }
        print(paste('parameters are:',bdd,'-',weight,'-',metric, sep = ""))

        
        
        
        
        #####################################################################################
        #5.3. model with layers and interaction with complexity
        #####################################################################################
        
        model_int_complexity = step(lm(rate ~ 
                                       +conv1_1+conv1_2+pool1   
                                       +conv2_1+conv2_2+pool2   
                                       +conv3_1+conv3_2+conv3_3+pool3   
                                       +conv4_1+conv4_2+conv4_3+pool4   
                                       +conv5_1+conv5_2+conv5_3+pool5
                                       ,data = df_metrics), trace=0)
        
       print(summary(model_int_complexity))
        
    




