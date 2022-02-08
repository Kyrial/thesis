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
setwd("/home/nicolas/Bureau/internship_cefe_2021/code/functions")
#####################################################################################
# 3. PARAMETRES: def analyse_metrics(model_name, bdd, weight, metric,k):
#####################################################################################
#####################################################################################
# 3.1 Parametres
#####################################################################################
#mettre ça pas en dur a terme mais en paramètres passé au script python (ou pas?)

model_name <- 'VGG16'
bdds <- c('MART','JEN','CFD','SCUT-FBP')
weights <- c('vggface','imagenet','vggplaces')
metrics <- c('gini_flatten')


main_model(model_name, bdds, weights, metrics)
tsn = as.data.frame(tsn)
plot(tsn$V1, tsn$V2)

#c1
res.pca <- PCA(df[,c(4,7,10,14,18)], graph = TRUE)
#c2
res.pca <- PCA(df[,c(5,8,11,15,19)], graph = TRUE)
#pool
res.pca <- PCA(df[,c(6,9,13,17,21)], graph = TRUE)




main_model = function(model_name, bdds, weights, metrics) {
  
  
  
  for (bdd in bdds) {
    for (weight in weights){
      for (metric in metrics){
        
        #####################################################################################
        # 3.2. Data management
        #####################################################################################
        if (model_name == 'VGG16'){
          
          if (weight == 'imagenet'){
            layers = c('input_1',
                       'block1_conv1','block1_conv2','block1_pool',
                       'block2_conv1','block2_conv2','block2_pool',
                       'block3_conv1','block3_conv2','block3_conv3','block3_pool',
                       'block4_conv1','block4_conv2','block4_conv3','block4_pool',
                       'block5_conv1','block5_conv2','block5_conv3','block5_pool',
                       'flatten','fc1', 'fc2')} 
          
          else if (weight == 'vggface'){
            layers = c('input_1',
                       'conv1_1','conv1_2','pool1',
                       'conv2_1','conv2_2','pool2',
                       'conv3_1','conv3_2','conv3_3','pool3',
                       'conv4_1','conv4_2','conv4_3','pool4',
                       'conv5_1','conv5_2','conv5_3','pool5',
                       'flatten','fc6/relu','fc7/relu')}
          
          else if (weight == 'vggplaces'){
            layers = c('input_1',
                       'block1_conv1','block1_conv2','block1_pool',
                       'block2_conv1','block2_conv2','block2_pool',
                       'block3_conv1','block3_conv2','block3_conv3','block3_pool',
                       'block4_conv1','block4_conv2','block4_conv3','block4_pool',
                       'block5_conv1','block5_conv2','block5_conv3','block5_pool',
                       'flatten','fc1', 'fc2')}}
        
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
        
        matrix_complexity <- do.call(cbind, fromJSON(file = paste(log_path,'_',bdd,'_',weight,'_','mean','_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
        colnames(matrix_complexity)[2] <- 'input_1'
        
        
        
        if (metric %in% c('kurtosis','L0')){
          
          df_metrics <- as.data.frame(matrix_metrics, optional = TRUE)
          flatten_df_metrics = flatten(df_metrics[,-1])
          vmin = min(unlist(flatten_df_metrics))
          vmax = max(unlist(flatten_df_metrics))
          df_metrics = sapply(df_metrics, as.numeric)
          df_metrics[,-1]= df_metrics[,-1] - vmin
          df_metrics[,-1]= df_metrics[,-1] / vmax
          
        }else{ 
          
          df_metrics <- as.data.frame(matrix_metrics, optional = TRUE)
          df_metrics = sapply(df_metrics, as.numeric)
        }
        
        df_complexity <- as.data.frame(matrix_complexity, optional = TRUE)
        df_complexity = sapply(df_complexity, as.numeric)
        
        
        df_metrics <- as.data.frame(df_metrics)
        df_complexity <- as.data.frame(df_complexity[,-1])
        
        df_sq_metrics = df_metrics[,-1]*df_metrics[,-1]
        
        if (weight == 'imagenet'){
          df_sq_metrics = rename(df_sq_metrics, c("input_1" = "input_1_sq",
                                                  'block1_conv1'='conv1_1_sq','block1_conv2'='conv1_2_sq','block1_pool'='pool1_sq',
                                                  'block2_conv1'='conv2_1_sq','block2_conv2'='conv2_2_sq','block2_pool'='pool2_sq',
                                                  'block3_conv1'='conv3_1_sq','block3_conv2'='conv3_2_sq','block3_conv3'='conv3_3_sq','block3_pool'='pool3_sq',
                                                  'block4_conv1'='conv4_1_sq','block4_conv2'='conv4_2_sq','block4_conv3'='conv4_3_sq','block4_pool'='pool4_sq',
                                                  'block5_conv1'='conv5_1_sq','block5_conv2'='conv5_2_sq','block5_conv3'='conv5_3_sq','block5_pool'='pool5_sq',
                                                  'flatten'='flatten_sq','fc1'='fc6/relu_sq','fc2'='fc7/relu_sq'))
        }else if (weight == 'vggface'){
          df_sq_metrics = rename(df_sq_metrics, c('input_1'='input_1_sq',
                                                  'conv1_2'='conv1_2_sq','conv1_1'='conv1_1_sq','pool1'='pool1_sq',
                                                  'conv2_1'='conv2_1_sq','conv2_2'='conv2_2_sq','pool2'='pool2_sq',
                                                  'conv3_1'='conv3_1_sq','conv3_2'='conv3_2_sq','conv3_3'='conv3_3_sq','pool3'='pool3_sq',
                                                  'conv4_1'='conv4_1_sq','conv4_2'='conv4_2_sq','conv4_3'='conv4_3_sq','pool4'='pool4_sq',
                                                  'conv5_1'='conv5_1_sq','conv5_2'='conv5_2_sq','conv5_3'='conv5_3_sq','pool5'='pool5_sq',
                                                  'flatten'='flatten_sq','fc6/relu'='fc6/relu_sq','fc7/relu'='fc7/relu_sq'))
        }else if (weight == 'vggplaces'){
          df_sq_metrics = rename(df_sq_metrics, c("input_1" = "input_1_sq",
                                                  'block1_conv1'='conv1_1_sq','block1_conv2'='conv1_2_sq','block1_pool'='pool1_sq',
                                                  'block2_conv1'='conv2_1_sq','block2_conv2'='conv2_2_sq','block2_pool'='pool2_sq',
                                                  'block3_conv1'='conv3_1_sq','block3_conv2'='conv3_2_sq','block3_conv3'='conv3_3_sq','block3_pool'='pool3_sq',
                                                  'block4_conv1'='conv4_1_sq','block4_conv2'='conv4_2_sq','block4_conv3'='conv4_3_sq','block4_pool'='pool4_sq',
                                                  'block5_conv1'='conv5_1_sq','block5_conv2'='conv5_2_sq','block5_conv3'='conv5_3_sq','block5_pool'='pool5_sq',
                                                  'flatten'='flatten_sq','fc1'='fc6/relu_sq','fc2'='fc7/relu_sq'))}
        
        df_sq_metrics <- cbind(df_metrics, df_sq_metrics)
        
        
        #####################################################################################
        #4. COMPLEXITY (saraee et all's paper)
        #####################################################################################
        
        df_complexity = rename(df_complexity, c("input_1" = "input_1",
                                                'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                                'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                                'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                                'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                                'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                                'flatten'='flatten','fc1'='fc6/relu','fc2'='fc7/relu'))
        
        df_complexity_metrics <- cbind(df_complexity$conv4_3, df_metrics)
        df_complexity_metrics = rename(df_complexity_metrics, c('df_complexity$conv4_3'='complexity') )
        
        df_birkhoff = df_metrics[,-1:-2]/df_complexity$conv4_3
        df_birkhoff = cbind(df_metrics$rate, df_birkhoff) 
        df_birkhoff = rename(df_birkhoff, c('df_metrics$rate'='rate') )
        
        df_birkhoff2 = df_metrics[,-1:-2]*df_complexity$conv4_3
        df_birkhoff2 = cbind(df_metrics$rate, df_birkhoff2) 
        df_birkhoff2 = rename(df_birkhoff2, c('df_metrics$rate'='rate') )
        #####################################################################################
        # 5. MULTIPLES MODELS
        #####################################################################################
        
        #faire des modèles avec les métriques intermédiaires (points d'inflexion), et des interactions comme la complexité par ex)
        if (weight %in% c('imagenet','vggplaces')) {
          df_metrics = rename(df_metrics, c("input_1" = "input_1",
                                            'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                            'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                            'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                            'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                            'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                            'flatten'='flatten','fc1'='fc6/relu','fc2'='fc7/relu'))
          
          df_complexity_metrics = rename(df_complexity_metrics, c("input_1" = "input_1",
                                                                  'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                                                  'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                                                  'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                                                  'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                                                  'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                                                  'flatten'='flatten','fc1'='fc6/relu','fc2'='fc7/relu'))
          
          df_birkhoff2 = rename(df_birkhoff2, c("input_1" = "input_1",
                                                'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                                'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                                'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                                'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                                'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                                'flatten'='flatten','fc1'='fc6/relu','fc2'='fc7/relu'))
          df_birkhoff = rename(df_birkhoff, c("input_1" = "input_1",
                                              'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                              'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                              'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                              'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                              'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                              'flatten'='flatten','fc1'='fc6/relu','fc2'='fc7/relu'))
          
        }
        print(paste('parameters are:',bdd,'-',weight,'-',metric, sep = ""))
        ###########################
        #Z-transformation
        ###########################
        df_complexity_metrics$complexity <- scale(df_complexity_metrics$complexity)
        df_complexity_metrics$input_1 <- scale(df_complexity_metrics$input_1)
        df_complexity_metrics$conv1_1 <- scale(df_complexity_metrics$conv1_1)
        df_complexity_metrics$conv1_2 <- scale(df_complexity_metrics$conv1_2)
        df_complexity_metrics$pool1 <- scale(df_complexity_metrics$pool1)
        df_complexity_metrics$conv2_1 <- scale(df_complexity_metrics$conv2_1)
        df_complexity_metrics$conv2_2 <- scale(df_complexity_metrics$conv2_2 )
        df_complexity_metrics$pool2 <- scale(df_complexity_metrics$pool2)
        df_complexity_metrics$conv3_1 <- scale(df_complexity_metrics$conv3_1)
        df_complexity_metrics$conv3_2 <- scale(df_complexity_metrics$conv3_2)
        df_complexity_metrics$conv3_3 <- scale(df_complexity_metrics$conv3_3)
        df_complexity_metrics$pool3 <- scale(df_complexity_metrics$pool3)
        df_complexity_metrics$conv4_1 <- scale(df_complexity_metrics$conv4_1)
        df_complexity_metrics$conv4_2 <- scale(df_complexity_metrics$conv4_2)
        df_complexity_metrics$conv4_3 <- scale(df_complexity_metrics$conv4_3)
        df_complexity_metrics$pool4<- scale(df_complexity_metrics$pool4)
        df_complexity_metrics$conv5_1 <- scale(df_complexity_metrics$conv5_1)
        df_complexity_metrics$conv5_2 <- scale(df_complexity_metrics$conv5_2)
        df_complexity_metrics$conv5_3 <- scale(df_complexity_metrics$conv5_3)
        df_complexity_metrics$pool5<- scale(df_complexity_metrics$pool5)
        
        
        
        #calcul des r de pearson par couche
        
        correlations <- apply(df_complexity_metrics[,c(1,3:21)],2,cor, y = df_complexity_metrics$rate )
        
        blocks_cols = c('black','grey','pink','pink','pink',
                        'blue','blue','blue','red','red','red','red',
                        'green','green','green','green','yellow','yellow','yellow','yellow')
        
        path = gsub(" ","",paste('../../results/',bdd,'/barplots/'))
        
        jpeg(gsub(" ","",paste(path,'R_pearson'," ",bdd," ",weight," ",metric,".jpg")), width = 800, height = 800)
        barplot(correlations, col = blocks_cols, main = paste("R Pearson"," ",bdd," ",weight," ",metric),
                xlab = "layer", ylab = "R")
        dev.off()
        
        jpeg(gsub(" ","",paste(path,"R2_pearson "," ",bdd," ",weight," ",metric,".jpg")), width = 800, height = 800)
        barplot(correlations*correlations, col= blocks_cols, main = paste("R2 pearson "," ",bdd," ",weight," ",metric),
                xlab = "layer", ylab = "R2")
        dev.off()
        
        
        
        #####################################################################################
        #5.3. model with layers and interaction with complexity
        #####################################################################################
        
        model_int_complexity = step(lm(rate ~ 
                                         (
                                           conv1_2+conv1_1+pool1   
                                           +conv2_1+conv2_2+pool2   
                                           +conv3_1+conv3_2+conv3_3+pool3   
                                           +conv4_1+conv4_2+conv4_3+pool4   
                                           +conv5_1+conv5_2+conv5_3+pool5
                                           +complexity):complexity
                                       
                                       +conv1_1+conv1_2+pool1   
                                       +conv2_1+conv2_2+pool2   
                                       +conv3_1+conv3_2+conv3_3+pool3   
                                       +conv4_1+conv4_2+conv4_3+pool4   
                                       +conv5_1+conv5_2+conv5_3+pool5
                                       ,data = df_complexity_metrics), trace=0)
        
        #print(summary(model_int_complexity))
        
        #####################################################################################
        #5.4 corrélations simples
        #####################################################################################
        
        #subset avec les dernieres couches de conv
        
        df_lastconv = df_complexity_metrics[,c("rate","conv1_2","conv2_2","conv3_3","conv4_3","conv5_3","complexity")]
        
        #attractvité/sp max
        df_lastconv$sp_max = do.call(pmax, df_lastconv[,c("conv1_2","conv2_2","conv3_3","conv4_3","conv5_3")])
        df_lastconv$sp_min = do.call(pmin, df_lastconv[,c("conv1_2","conv2_2","conv3_3","conv4_3","conv5_3")])
        df_lastconv$scope = df_lastconv$sp_max - df_lastconv$sp_min
        #df_lastconv$mean = rowMeans(df_lastconv["conv1_2","conv2_2","conv3_3","conv4_3","conv5_3"])
        df_lastconv$mean = rowMeans(df_lastconv[,c("conv1_2","conv2_2","conv3_3","conv4_3","conv5_3")])
        
        spmax = lm(rate~sp_max, data = df_lastconv)
        #plot(df_lastconv$rate,df_lastconv$sp_max)
        print("1:")
        r_one = cor(df_lastconv$conv1_2, df_lastconv$rate, method = "pearson")
        rc_one = cor(df_lastconv$conv1_2, df_lastconv$rate, method = "pearson")^2 
        print("2:")
        r_two = cor(df_lastconv$conv2_2, df_lastconv$rate, method = "pearson")
        rc_two = cor(df_lastconv$conv2_2, df_lastconv$rate, method = "pearson")^2 
        print("3:")
        r_three = cor(df_lastconv$conv3_3, df_lastconv$rate, method = "pearson")
        rc_three = cor(df_lastconv$conv3_3, df_lastconv$rate, method = "pearson")^2 
        print("4:")
        r_four = cor(df_lastconv$conv4_3, df_lastconv$rate, method = "pearson")
        rc_four = cor(df_lastconv$conv4_3, df_lastconv$rate, method = "pearson")^2 
        print("5:")
        r_five = cor(df_lastconv$conv5_3, df_lastconv$rate, method = "pearson")
        rc_five = cor(df_lastconv$conv5_3, df_lastconv$rate, method = "pearson")^2 
        
        model_multi = step(lm(rate ~ conv1_2+conv2_2+conv3_3+conv4_3+conv5_3 + complexity + (conv1_2+conv2_2+conv3_3+conv4_3+conv5_3):complexity
                              ,data = df_lastconv), trace=0)
        #plot(model_multi)
        #print(summary(model_multi))
        
        barplot(c(r_one,r_two,r_three,r_four,r_five), main = "corrélation")
        barplot(c(rc_one,rc_two,rc_three,rc_four,rc_five),main = "variance expliquée")
      } 
    }
  }
  graphics.off()
}







