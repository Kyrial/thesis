#####################################################################################
# 1. DESCRIPTION:
#####################################################################################

#Modèle sparsité+complexité par couche, avec des régressions régularisées (ridge et lasso), en cross validation (10-fold)

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
library("glmnet")
library(data.table)
setwd("/home/renoult/Bureau/thesis/code/functions")
#####################################################################################
# 3. PARAMETERS:
#####################################################################################
model_name <- 'VGG16'
bdd <- c('JEN')
weight <- c('imagenet')
metric <- c('gini_flatten')
regularization <- 'ridge' #ridge for ridge, lasso for lasso, glmnet for elasticnet
#####################################################################################
# 4. DATA MANAGEMENT
#####################################################################################
labels_path = paste('../../data/redesigned/',bdd,'/labels_',bdd,'.csv', sep="")
log_path =paste('../../results/',bdd,'/log_', sep="")

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

#changement des noms de colonne pour les uniformiser car les differents weights ont des noms de layers différents
df_metrics = plyr::rename(df_metrics, c("input_1" = "input_1",
                                        'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                        'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                        'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                        'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                        'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                        'flatten'='flatten','fc1'='fc6_relu','fc2'='fc7_relu'))
#même démarche pour la complexité
df_complexity = plyr::rename(df_complexity, c("input_1" = "input_1_comp",
                                              'block1_conv1'='conv1_1_comp','block1_conv2'='conv1_2_comp','block1_pool'='pool1_comp',
                                              'block2_conv1'='conv2_1_comp','block2_conv2'='conv2_2_comp','block2_pool'='pool2_comp',
                                              'block3_conv1'='conv3_1_comp','block3_conv2'='conv3_2_comp','block3_conv3'='conv3_3_comp','block3_pool'='pool3_comp',
                                              'block4_conv1'='conv4_1_comp','block4_conv2'='conv4_2_comp','block4_conv3'='conv4_3_comp','block4_pool'='pool4_comp',
                                              'block5_conv1'='conv5_1_comp','block5_conv2'='conv5_2_comp','block5_conv3'='conv5_3_comp','block5_pool'='pool5_comp',
                                              'flatten'='flatten_comp','fc1'='fc6_relu_comp','fc2'='fc7_relu_comp'))

#on vire les colonnes des couches d'input, de pooling fc et flatten (on garde les couches de convolution)
df_metrics = subset(df_metrics, select=c(  'rate',
                                           'conv1_1','conv1_2',
                                           'conv2_1','conv2_2',
                                           'conv3_1','conv3_2','conv3_3',
                                           'conv4_1','conv4_2','conv4_3',
                                           'conv5_1','conv5_2','conv5_3'
                                         ))

df_complexity = subset(df_complexity, select=c(  'conv1_1_comp','conv1_2_comp',
                                                 'conv2_1_comp','conv2_2_comp',
                                                 'conv3_1_comp','conv3_2_comp','conv3_3_comp',
                                                 'conv4_1_comp','conv4_2_comp','conv4_3_comp',
                                                 'conv5_1_comp','conv5_2_comp','conv5_3_comp'
                                              ))


#création d'un dataframe avec la complexité, la sparsité approximée par gini
df <- cbind(df_metrics, df_complexity)

#Z-transformation (centré réduit)
scaled_df <- scale(df[,-1]) #df[,-1] pour ne pas z transformer la beauté
df <- cbind(df$rate ,scaled_df) #si on avait pas scaled la beauté il aurait fallu la remettre
#df = scaled_df
df<- as.data.frame(df, optional = TRUE) #ce n'est plus un  dataframe, il faut refaire en sorte que ça en soit un
df <- plyr::rename(df, c("V1" = "rate")) #la colonne des notes ne s'apellelait plus "rate"

#complexité avec la note
df_complexity_rate = cbind(df$rate ,df_complexity)
df_complexity_rate<- as.data.frame(df_complexity_rate, optional = TRUE) 
df_complexity_rate <- plyr::rename(df_complexity_rate, c("df$rate" = "rate")) #la colonne des notes ne s'apellelait plus "rate"


k = nrow(matrix)

set.seed(123)

ctrl = trainControl(method = "cv", number = 10) #10-fold cv
##################################################################################
#6:Model
#####################################################################################


model = train( 
  rate ~ .,
  data = df,
  method = regularization,
  preProc = c("center", "scale"),
  trControl = ctrl,
  metric = "Rsquared")

r_squared = model$results$Rsquared[1]

print(r_squared)

######################################
#6:Effect size
######################################
#for comp only

effect_size = varImp(model, scale = FALSE)
effect_size = effect_size$importance

comp1_1 = effect_size[14,]
comp1_2 = effect_size[15,]
comp2_1 = effect_size[16,]
comp2_2 = effect_size[17,]
comp3_1 = effect_size[18,]
comp3_2 = effect_size[19,]
comp3_3 = effect_size[20,]
comp4_1 = effect_size[21,]
comp4_2 = effect_size[22,]
comp4_3 = effect_size[23,]
comp5_1 = effect_size[24,]
comp5_2 = effect_size[25,]
comp5_3 = effect_size[26,]

xlab = c( 'comp1_1','comp1_2', 
          'comp2_1','comp2_2',
          'comp3_1','comp3_2', 'comp3_3',
          'comp4_1','comp4_2', 'comp4_3',
          'comp5_1','comp5_2', 'comp5_3' )

effect_size_order = c( comp1_1,comp1_2, 
                       comp2_1,comp2_2,
                       comp3_1,comp3_2, comp3_3,
                       comp4_1,comp4_2, comp4_3,
                       comp5_1,comp5_2, comp5_3 )

barplot(effect_size_order, names = xlab, las = 2, col = "red", main = paste('Effect size for ',bdd,' with ',regularization," regularization" ,", complexity only",sep=""), ylim=c(0,0.20))





#for Sp + comp
effect_size = varImp(model, scale = FALSE)
effect_size = effect_size$importance

spars1_1 = effect_size[1,]
spars1_2 = effect_size[2,]
spars2_1 = effect_size[3,]
spars2_2 = effect_size[4,]
spars3_1 = effect_size[5,]
spars3_2 = effect_size[6,]
spars3_3 = effect_size[7,]
spars4_1 = effect_size[8,]
spars4_2 = effect_size[9,]
spars4_3 = effect_size[10,]
spars5_1 = effect_size[11,]
spars5_2 = effect_size[12,]
spars5_3 = effect_size[13,]

comp1_1 = effect_size[14,]
comp1_2 = effect_size[15,]
comp2_1 = effect_size[16,]
comp2_2 = effect_size[17,]
comp3_1 = effect_size[18,]
comp3_2 = effect_size[19,]
comp3_3 = effect_size[20,]
comp4_1 = effect_size[21,]
comp4_2 = effect_size[22,]
comp4_3 = effect_size[23,]
comp5_1 = effect_size[24,]
comp5_2 = effect_size[25,]
comp5_3 = effect_size[26,]


xlab = c( 'spars1_1','comp1_1', 'spars1_2','comp1_2', 
          'spars2_1','comp2_1', 'spars2_2','comp2_2',
          'spars3_1','comp3_1', 'spars3_2','comp3_2', 'spars3_3','comp3_3',
          'spars4_1','comp4_1', 'spars4_2','comp4_2', 'spars4_3','comp4_3',
          'spars5_1','comp5_1', 'spars5_2','comp5_2', 'spars5_3','comp5_3' )

effect_size_order = c( spars1_1,comp1_1, spars1_2,comp1_2, 
                       spars2_1,comp2_1, spars2_2,comp2_2,
                       spars3_1,comp3_1, spars3_2,comp3_2, spars3_3,comp3_3,
                       spars4_1,comp4_1, spars4_2,comp4_2, spars4_3,comp4_3,
                       spars5_1,comp5_1, spars5_2,comp5_2, spars5_3,comp5_3 )

colors = c("yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red")

barplot(effect_size_order, names = xlab, las = 2, col = colors, main = paste('Effect size for ',bdd,' with ',regularization," regularization" ,", sparseness + complexity without interaction",sep=""), ylim=c(0,0.20))


#for Sp only
effect_size = varImp(model, scale = FALSE)
effect_size = effect_size$importance

spars1_1 = effect_size[1,]
spars1_2 = effect_size[2,]
spars2_1 = effect_size[3,]
spars2_2 = effect_size[4,]
spars3_1 = effect_size[5,]
spars3_2 = effect_size[6,]
spars3_3 = effect_size[7,]
spars4_1 = effect_size[8,]
spars4_2 = effect_size[9,]
spars4_3 = effect_size[10,]
spars5_1 = effect_size[11,]
spars5_2 = effect_size[12,]
spars5_3 = effect_size[13,]

xlab = c( 'spars1_1', 'spars1_2',
          'spars2_1', 'spars2_2',
          'spars3_1', 'spars3_2','spars3_3',
          'spars4_1', 'spars4_2','spars4_3',
          'spars5_1', 'spars5_2','spars5_3')

effect_size_order = c( spars1_1, spars1_2, 
                       spars2_1, spars2_2,
                       spars3_1, spars3_2, spars3_3,
                       spars4_1, spars4_2, spars4_3,
                       spars5_1, spars5_2, spars5_3 )

barplot(effect_size_order, names = xlab, las = 2, col = "yellow" , main = paste('Effect size for ',bdd,' with ',regularization," regularization" ,", sparseness only",sep=""), ylim=c(0,0.20))





  
#################################################################################"

#ACHIVES:

#####################################################################################
# 5. MODEL: RIDGE/LASSO REGRESSION avec interaction
#####################################################################################
model_inter = train( rate ~ 
                       conv1_1+conv1_2+conv2_1+conv2_2+conv3_1+conv3_2+conv3_3+conv4_1+conv4_2+conv4_3+conv5_1+conv5_2+conv5_3+ #sparsité
                       
                       conv1_1_comp+conv1_2_comp+conv2_1_comp+conv2_2_comp+conv3_1_comp+conv3_2_comp+conv3_3_comp+conv4_1_comp+conv4_2_comp+
                       conv4_3_comp+conv5_1_comp+conv5_2_comp+conv5_3_comp+ #complexité
                       
                       conv1_1:conv1_1_comp+conv1_2:conv1_2_comp+
                       conv2_1:conv2_1_comp+conv2_2:conv2_2_comp+
                       conv3_1:conv3_1_comp+conv3_2:conv3_2_comp+conv3_3:conv3_3_comp+
                       conv4_1:conv4_1_comp+conv4_2:conv4_2_comp+conv4_3:conv4_3_comp+
                       conv5_1:conv5_1_comp+conv5_2:conv5_2_comp+conv5_3:conv5_3_comp #interactions entre les deux, par couche
                     
                     ,data = df ,method = regularization,preProc = c("center", "scale"),trControl = ctrl, metric = "Rsquared")

r_squared = model_inter$results$Rsquared[1]

print(r_squared)






