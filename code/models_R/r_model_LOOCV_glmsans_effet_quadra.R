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
library("caret")
library("jtools")
library("broom.mixed")
setwd("/home/renoult/Bureau/internship_cefe_2021/code/functions")
#####################################################################################
# 3. PARAMETERS:
#####################################################################################
model_name <- 'VGG16'
bdd <- c('SCUT-FBP')
weight <- c('imagenet')
metric <- c('gini_flatten')
#####################################################################################
# 4. DATA MANAGEMENT
#####################################################################################
labels_path ='../../data/redesigned/SCUT-FBP/labels_SCUT-FBP.csv'
log_path ='../../results/SCUT-FBP/log_'

#chargement du fichier
matrix_metrics <- do.call(cbind, fromJSON(file = paste(log_path,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
      
#si on ne fait pas ça, l'input peut avoir un indice variable
colnames(matrix_metrics)[2] <- 'input_1'
        
#idem avec les calculs de complexité
matrix_complexity <- do.call(cbind, fromJSON(file = paste(log_path,'_',bdd,'_',weight,'_','mean','_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
colnames(matrix_complexity)[2] <- 'input_1'
          
#passage des matrice en dataframe
df_metrics <- as.data.frame(matrix_metrics, optional = TRUE)
df_complexity <- as.data.frame(matrix_complexity, optional = TRUE)        
#passage en flottants (avant c'était des strings)
df_metrics = sapply(df_metrics, as.numeric)
df_complexity = sapply(df_complexity, as.numeric)
        
#il faut repasser en df après le sapply
df_metrics <- as.data.frame(df_metrics)
df_complexity <- as.data.frame(df_complexity[,-1])
        
#effets quadratique de la sparseness
df_temp = df_metrics[,-1]
df_sq_metrics = df_temp*df_temp

#effets quadratiques de la complexité
df_sq_complexity = df_complexity*df_complexity       

#changement des noms de colonne pour les uniformiser car les differents weights ont des noms de layers différents
df_metrics = plyr::rename(df_metrics, c("input_1" = "input_1",
                                         'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                         'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                         'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                         'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                         'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                         'flatten'='flatten','fc1'='fc6_relu','fc2'='fc7_relu'))
df_sq_metrics = plyr::rename(df_sq_metrics, c("input_1" = "input_1_sq",
                                              'block1_conv1'='conv1_1_sq','block1_conv2'='conv1_2_sq','block1_pool'='pool1_sq',
                                              'block2_conv1'='conv2_1_sq','block2_conv2'='conv2_2_sq','block2_pool'='pool2_sq',
                                              'block3_conv1'='conv3_1_sq','block3_conv2'='conv3_2_sq','block3_conv3'='conv3_3_sq','block3_pool'='pool3_sq',
                                              'block4_conv1'='conv4_1_sq','block4_conv2'='conv4_2_sq','block4_conv3'='conv4_3_sq','block4_pool'='pool4_sq',
                                              'block5_conv1'='conv5_1_sq','block5_conv2'='conv5_2_sq','block5_conv3'='conv5_3_sq','block5_pool'='pool5_sq',
                                              'flatten'='flatten_sq','fc1'='fc6_relu_sq','fc2'='fc7_relu_sq'))
#même démarche pour la complexité
df_complexity = plyr::rename(df_complexity, c("input_1" = "input_1_comp",
                                              'block1_conv1'='conv1_1_comp','block1_conv2'='conv1_2_comp','block1_pool'='pool1_comp',
                                              'block2_conv1'='conv2_1_comp','block2_conv2'='conv2_2_comp','block2_pool'='pool2_comp',
                                              'block3_conv1'='conv3_1_comp','block3_conv2'='conv3_2_comp','block3_conv3'='conv3_3_comp','block3_pool'='pool3_comp',
                                              'block4_conv1'='conv4_1_comp','block4_conv2'='conv4_2_comp','block4_conv3'='conv4_3_comp','block4_pool'='pool4_comp',
                                              'block5_conv1'='conv5_1_comp','block5_conv2'='conv5_2_comp','block5_conv3'='conv5_3_comp','block5_pool'='pool5_comp',
                                              'flatten'='flatten_comp','fc1'='fc6_relu_comp','fc2'='fc7_relu_comp'))

#et pour les effts quadratiques de la complexité
df_sq_complexity = plyr::rename(df_sq_complexity, c("input_1" = "input_1_comp_sq",
                                              'block1_conv1'='conv1_1_comp_sq','block1_conv2'='conv1_2_comp_sq','block1_pool'='pool1_comp_sq',
                                              'block2_conv1'='conv2_1_comp_sq','block2_conv2'='conv2_2_comp_sq','block2_pool'='pool2_comp_sq',
                                              'block3_conv1'='conv3_1_comp_sq','block3_conv2'='conv3_2_comp_sq','block3_conv3'='conv3_3_comp_sq','block3_pool'='pool3_comp_sq',
                                              'block4_conv1'='conv4_1_comp_sq','block4_conv2'='conv4_2_comp_sq','block4_conv3'='conv4_3_comp_sq','block4_pool'='pool4_comp_sq',
                                              'block5_conv1'='conv5_1_comp_sq','block5_conv2'='conv5_2_comp_sq','block5_conv3'='conv5_3_comp_sq','block5_pool'='pool5_comp_sq',
                                              'flatten'='flatten_comp_sq','fc1'='fc6_relu_comp_sq','fc2'='fc7_relu_comp_sq'))

#création d'un dataframe avec la complexité, la sparsité approximée par gini et les effets quadratiques de la sparsité
df <- cbind(df_metrics, df_sq_metrics, df_complexity, df_sq_complexity)
        
#Z-transformation (centré réduit)
scaled_df <- scale(df) #df[,-1] pour ne pas z transformer la beauté
df <- cbind(df$rate ,scaled_df)
df<- as.data.frame(df, optional = TRUE)
df <- plyr::rename(df, c("V1" = "rate"))

#####################################################################################
# 5. MODEL: LEAVE ONE OUT CROSS VALIDATION
#####################################################################################

#on va itérer sur les couches, il faut donc leurs noms
layers = c('input_1','conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2',
'conv3_1','conv3_2','conv3_3','pool3','conv4_1','conv4_2','conv4_3','pool4',
'conv5_1','conv5_2','conv5_3','pool5','flatten','fc6/relu','fc7/relu')


ctrl <- trainControl(method = "LOOCV")

#input
model_input_1 <- train(rate ~ 
                 input_1 + input_1_comp + (input_1 * input_1_comp) ,
               data = df, method = "lm", trControl = ctrl)
input_1 <- c(model_input_1$results$Rsquared)

#conv1_1
model_conv1_1 <- train(rate ~ 
                 conv1_1 + conv1_1_comp + (conv1_1 * conv1_1_comp)  ,
               data = df, method = "lm", trControl = ctrl)
conv1_1 <- c(model_conv1_1$results$Rsquared)

#conv1_2
model_conv1_2 <- train(rate ~ 
                 conv1_2 + conv1_2_comp + (conv1_2 * conv1_2_comp)  ,
               data = df, method = "lm", trControl = ctrl)
conv1_2 <- c(model_conv1_2$results$Rsquared)

#pool1
model_pool1 <- train(rate ~ 
                 pool1 + pool1_comp + (pool1 * pool1_comp) ,
               data = df, method = "lm", trControl = ctrl)
pool1 <- c(model_pool1$results$Rsquared)

#conv2_1
model_conv2_1 <- train(rate ~ 
                 conv2_1 + conv2_1_comp + (conv2_1 * conv2_1_comp)  ,
               data = df, method = "lm", trControl = ctrl)
conv2_1 <- c(model_conv2_1$results$Rsquared)

#conv2_2
model_conv2_2 <- train(rate ~ 
                 conv2_2 + conv2_2_comp + (conv2_2 * conv2_2_comp)  ,
               data = df, method = "lm", trControl = ctrl)
conv2_2 <- c(model_conv2_2$results$Rsquared)

#pool2
model_pool2 <- train(rate ~ 
                 pool2 + pool2_comp + (pool2 * pool2_comp) ,
               data = df, method = "lm", trControl = ctrl)
pool2  <- c(model_pool2$results$Rsquared)

#conv3_1
model_conv3_1 <- train(rate ~ 
                 conv3_1 + conv3_1_comp + (conv3_1 * conv3_1_comp) ,
               data = df, method = "lm", trControl = ctrl)

conv3_1 <-c(model_conv3_1$results$Rsquared)

#conv3_2
model_conv3_2 <- train(rate ~ 
                 conv3_2 + conv3_2_comp + (conv3_2 * conv3_2_comp),
               data = df, method = "lm", trControl = ctrl)
conv3_2 <-c(model_conv3_2$results$Rsquared)

#conv3_3
model_conv3_3 <- train(rate ~ 
                 conv3_3 + conv3_3_comp + (conv3_3 * conv3_3_comp) ,
               data = df, method = "lm", trControl = ctrl)
conv3_3 <-c(model_conv3_3$results$Rsquared)

#pool3
model_pool3 <- train(rate ~ 
                 pool3 + pool3_comp + (pool3 * pool3_comp)  ,
               data = df, method = "lm", trControl = ctrl)
pool3 <-c(model_pool3$results$Rsquared)

#conv4_1
model_conv4_1 <- train(rate ~ 
                 conv4_1 + conv4_1_comp + (conv4_1 * conv4_1_comp),
               data = df, method = "lm", trControl = ctrl)
conv4_1 <-c(model_conv4_1$results$Rsquared)

#conv4_2
model_conv4_2 <- train(rate ~ 
                 conv4_2 + conv4_2_comp + (conv4_2 * conv4_2_comp)  ,
               data = df, method = "lm", trControl = ctrl)
conv4_2 <-c(model_conv4_2$results$Rsquared)

#conv4_3
model_conv4_3 <- train(rate ~ 
                 conv4_3 + conv4_3_comp + (conv4_3 * conv4_3_comp) ,
               data = df, method = "lm", trControl = ctrl)
conv4_3 <-c(model_conv4_3$results$Rsquared)

#pool4
model_pool4 <- train(rate ~ 
                 pool4 + pool4_comp + (pool4 * pool4_comp) ,
               data = df, method = "lm", trControl = ctrl)
pool4 <-c(model_pool4$results$Rsquared)

#conv5_1
model_conv5_1 <- train(rate ~ 
                 conv5_1 + conv5_1_comp + (conv5_1 * conv5_1_comp)  ,
               data = df, method = "lm", trControl = ctrl)
conv5_1 <-c(model_conv5_1$results$Rsquared)

#conv5_2
model_conv5_2 <- train(rate ~ 
                 conv5_2 + conv5_2_comp + (conv5_2 * conv5_2_comp) ,
               data = df, method = "lm", trControl = ctrl)
conv5_2 <-c(model_conv5_2$results$Rsquared)

#conv5_3
model_conv5_3 <- train(rate ~ 
                 conv5_3 + conv5_3_comp + (conv5_3 * conv5_3_comp) ,
               data = df, method = "lm", trControl = ctrl)
conv5_3 <-c(model_conv5_3$results$Rsquared)

#pool5
model_pool5 <- train(rate ~ 
                 pool5 + pool5_comp + (pool5 * pool5_comp)  ,
               data = df, method = "lm", trControl = ctrl)
pool5 <-c(model_pool5$results$Rsquared)

#flatten
model_flatten <- train(rate ~ 
                 flatten + flatten_comp + (flatten * flatten_comp)  ,
               data = df, method = "lm", trControl = ctrl)
flatten <- c(model_flatten$results$Rsquared)

#fc6_relu
model_fc6_relu <- train(rate ~ 
                 fc6_relu + fc6_relu_comp + (fc6_relu * fc6_relu_comp)  ,
               data = df, method = "lm", trControl = ctrl)
fc6_relu <- c(model_fc6_relu$results$Rsquared)

#fc7_relu
model_fc7_relu <- train(rate ~ 
                 fc7_relu + fc7_relu_comp + (fc7_relu * fc7_relu_comp) ,
               data = df, method = "lm", trControl = ctrl)
fc7_relu <- c(model_fc7_relu$results$Rsquared)

#graphe de l'évolution des R2 de chaque modèle
r_sq = data.frame(input_1,conv1_1,conv1_2,pool1,conv2_1,conv2_2,pool2,
             conv3_1,conv3_2,conv3_3,pool3,conv4_1,conv4_2,conv4_3,pool4,
             conv5_1,conv5_2,conv5_3,pool5,flatten,fc6_relu,fc7_relu)
r_sq = t(r_sq)
plot(r_sq)
text(r_sq,labels=layers)

#visualisation des estimate
#abscisses standardisées
#bloc 1
plot_summs(model_input_1$finalModel,model_conv1_1$finalModel,model_conv1_2$finalModel,model_pool1$finalModel,
           scale = TRUE, model.names = c("input","conv1_1","conv1_2","pool_1"))
#bloc 2
plot_summs(model_conv1_2$finalModel,model_conv2_1$finalModel,model_conv2_2$finalModel,model_pool2$finalModel,
           scale = TRUE, model.names = c(" ","conv2_1","conv2_2","pool_2"))
#bloc 3
plot_summs(model_conv1_2$finalModel,model_conv3_1$finalModel,model_conv3_2$finalModel,model_conv3_3$finalModel,model_pool3$finalModel,
           scale = TRUE, model.names = c(" ","conv3_1","conv3_2","conv3_3","pool_3"))
#bloc 4
plot_summs(model_conv1_2$finalModel,model_conv4_1$finalModel,model_conv4_2$finalModel,model_conv4_3$finalModel,model_pool4$finalModel,
           scale = TRUE, model.names = c(" ","conv4_1","conv4_2","conv4_3","pool_4"))
#bloc 5
plot_summs(model_conv1_2$finalModel, model_conv5_1$finalModel,model_conv5_2$finalModel,model_conv5_3$finalModel,model_pool5$finalModel,
           scale = TRUE, model.names = c(" ","conv5_1","conv5_2","conv5_3","pool_5"))

