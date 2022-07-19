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
library("dplyr")
library("caret")
library("jtools")
library("broom.mixed")
library("glmnet")
setwd("/home/renoult/Bureau/thesis/code/functions")
#####################################################################################
# 3. PARAMETERS:
#####################################################################################
model_name <- 'VGG16'
bdds <- c('CFD','CFD_A','CFD_AF','CFD_AM','CFD_B','CFD_BF','CFD_BM','CFD_F','CFD_L','CFD_LF','CFD_LM','CFD_M','CFD_W','CFD_WF','CFD_WM')
weight <- c('imagenet')
metric <- c('gini_flatten')
regularization = 'ridge'


print(bdds)
#method <- c('FeatureMap')
#method <- c('featureMap')
method <- c('pca')
#method <- c('average')

subset_db1 = 700

for(bdd in bdds){ 
  #bdd <- c('CFD')
  #####################################################################################
  # 4. DATA MANAGEMENT
  #####################################################################################
  labels_path = paste('../../data/redesigned/CFD_ALL/labels_',bdd,'.csv', sep="")
  #labels_path = paste('../../data/redesigned/',bdd,'/labels_',bdd,'.csv', sep="")
  #log_path =paste('../../results/',bdd,'/log_', sep="")
  log_path =paste('../../results/',bdd,'/LLH_',method,'/LLH_',bdd,'_AllLLH.csv',sep = '')
  
  #chargement du fichier
  
  #####
  matrix_metrics <- read_csv(file=log_path)
  
  #si on ne fait pas ça, l'input peut avoir un indice variable
  colnames(matrix_metrics)[2] <- 'input_1'
  matrix_metrics <- matrix_metrics[,-1]
  matrix_beauty <- do.call(cbind,read.csv(file=labels_path, header=FALSE))
  colnames(matrix_beauty) <- c("img","rate")
  df_beauty <-subset(matrix_beauty, select = c(rate))
  
  #concate note beaute et llh
  #df = cbind(df_beauty$rate, matrix_metrics)
  df_metrics = cbind(df_beauty, matrix_metrics)
  #df_metrics <- df[,-2]
  
  #passage en flottants (avant c'était des strings)
  df_metrics = sapply(df_metrics, as.numeric)
  #il faut repasser en df après le sapply
  df_metrics <- as.data.frame(df_metrics)
  #####
  
  
  #changement des noms de colonne pour les uniformiser car les differents weights ont des noms de layers différents
  df_metrics = plyr::rename(df_metrics, c("input_1" = "input_1",
                                          'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                          'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                          'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                          'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                          'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                          'flatten'='flatten','fc1'='fc6_relu','fc2'='fc7_relu'))
  
  
  #création d'un dataframe avec la complexité, la sparsité approximée par gini
  #df <- cbind(df_metrics, df_complexity)
  
  #Z-transformation (centré réduit)
  #scaled_df <- scale(df[,-1]) #df[,-1] pour ne pas z transformer la beauté
  #df <- cbind(df$rate ,scaled_df) #si on avait pas scaled la beauté il aurait fallu la remettre
  #df = scaled_df
  #df<- as.data.frame(df, optional = TRUE)
  #df <- plyr::rename(df, c("V1" = "rate"))
  
  #sparsité seule
  df = df_metrics[,c(1,3,4,6,7,9,10,11,13,14,15,17,18,19,22,23)]
  
  
  #####################################################################################
  # 5. MODEL: RIDGE REGRESSION(ou lasso)
  #####################################################################################
  
  #10-fold
  ctrl = trainControl(method = "repeatedcv", number = 10, repeats = 10) #10-fold cv
  
  #estimation du lambda
  lambdas = 10^seq(2,-4,by=-0.1)
  # model1 = cv.glmnet(y = as.matrix(df$rate), x = as.matrix(df[,-1]), alpha = 0, lambda=lambdas)
  # plot(model1)
  # model1$lambda.min
  
  
  #"vrai" model
  model = train( rate ~ ., data = df ,method = "glmnet", tuneGrid = expand.grid(alpha = 0, lambda = lambdas),preProc = c("center", "scale"),trControl = ctrl, metric = "Rsquared") #alpha = 1 pour ridge (0 pour lasso)
  r_squared = model$results$Rsquared[1]
  
  
  print(bdd)
  print(method)
  print(r_squared)
}


print("bdd")









#####################################################################################
# 6. Tailles d'effet
#####################################################################################


coeffs = coef(model$finalModel, model$bestTune$lambda)


#que complexité
comp1_1_r = coeffs@x[2]
comp1_2_r = coeffs@x[3]
comp2_1_r = coeffs@x[4]
comp2_2_r = coeffs@x[5]
comp3_1_r = coeffs@x[6]
comp3_2_r = coeffs@x[7]
comp3_3_r = coeffs@x[8]
comp4_1_r = coeffs@x[9]
comp4_2_r = coeffs@x[10]
comp4_3_r = coeffs@x[11]
comp5_1_r = coeffs@x[12]
comp5_2_r = coeffs@x[13]
comp5_3_r = coeffs@x[14]
compfc6_r = coeffs@x[15]
compfc7_r = coeffs@x[16]


xlab = c( '1_1','1_2', 
          '2_1','2_2',
          '3_1','3_2', '3_3',
          '4_1','4_2', '4_3',
          '5_1','5_2', '5_3',
          'fc6','fc7')



effect_size_order = c( comp1_1_r,comp1_2_r, 
                         comp2_1_r,comp2_2_r,
                         comp3_1_r,comp3_2_r, comp3_3_r,
                         comp4_1_r,comp4_2_r, comp4_3_r,
                         comp5_1_r,comp5_2_r, comp5_3_r,
                         compfc6_r,compfc7_r
                       )

barplot(effect_size_order, names = xlab, las = 2, col = "red", main = paste('Effect size for ',bdd,' with ',regularization," regularization" ,", complexity only",sep=""))


#for Sp + comp
spars1_1_r = coeffs@x[2]
spars1_2_r = coeffs@x[3]
spars2_1_r = coeffs@x[4]
spars2_2_r = coeffs@x[5]
spars3_1_r = coeffs@x[6]
spars3_2_r = coeffs@x[7]
spars3_3_r = coeffs@x[8]
spars4_1_r = coeffs@x[9]
spars4_2_r = coeffs@x[11]
spars4_3_r = coeffs@x[11]
spars5_1_r = coeffs@x[12]
spars5_2_r = coeffs@x[13]
spars5_3_r = coeffs@x[14]
sparsfc6_r = coeffs@x[15]
sparsfc7_r = coeffs@x[16]


comp1_1_r = coeffs@x[17]
comp1_2_r = coeffs@x[18]
comp2_1_r = coeffs@x[19]
comp2_2_r = coeffs@x[20]
comp3_1_r = coeffs@x[21]
comp3_2_r = coeffs@x[22]
comp3_3_r = coeffs@x[23]
comp4_1_r = coeffs@x[24]
comp4_2_r = coeffs@x[25]
comp4_3_r = coeffs@x[26]
comp5_1_r = coeffs@x[27]
comp5_2_r = coeffs@x[28]
comp5_3_r = coeffs@x[29]
compfc6_r = coeffs@x[30]
compfc7_r = coeffs@x[31]


xlab = c( 'spars1_1','comp1_1', 'spars1_2','comp1_2', 
          'spars2_1','comp2_1', 'spars2_2','comp2_2',
          'spars3_1','comp3_1', 'spars3_2','comp3_2', 'spars3_3','comp3_3',
          'spars4_1','comp4_1', 'spars4_2','comp4_2', 'spars4_3','comp4_3',
          'spars5_1','comp5_1', 'spars5_2','comp5_2', 'spars5_3','comp5_3',
          'sparsfc6','compfc6','sparsfc7','compfc7')

effect_size_order = c( spars1_1_r,comp1_1_r, spars1_2_r,comp1_2_r, 
                       spars2_1_r,comp2_1_r, spars2_2_r,comp2_2_r,
                       spars3_1_r,comp3_1_r, spars3_2_r,comp3_2_r, spars3_3_r,comp3_3_r,
                       spars4_1_r,comp4_1_r, spars4_2_r,comp4_2_r, spars4_3_r,comp4_3_r,
                       spars5_1_r,comp5_1_r, spars5_2_r,comp5_2_r, spars5_3_r,comp5_3_r, 
                       sparsfc6_r,compfc6_r,sparsfc7_r,compfc7_r
                       )

colors = c("yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red","yellow","red")

barplot(effect_size_order, names = xlab, las = 2, col = colors, main = paste('Effect size for ',bdd,' with ',regularization," regularization" ,", sparseness + complexity without interaction",sep=""))

#for Sp only
spars1_1_r = coeffs@x[2]
spars1_2_r = coeffs@x[3]
spars2_1_r = coeffs@x[4]
spars2_2_r = coeffs@x[5]
spars3_1_r = coeffs@x[6]
spars3_2_r = coeffs@x[7]
spars3_3_r = coeffs@x[8]
spars4_1_r = coeffs@x[9]
spars4_2_r = coeffs@x[11]
spars4_3_r = coeffs@x[11]
spars5_1_r = coeffs@x[12]
spars5_2_r = coeffs@x[13]
spars5_3_r = coeffs@x[14]
sparsfc6_r = coeffs@x[15]
sparsfc7_r = coeffs@x[16]

xlab = c( 'spars1_1', 'spars1_2',
          'spars2_1', 'spars2_2',
          'spars3_1', 'spars3_2','spars3_3',
          'spars4_1', 'spars4_2','spars4_3',
          'spars5_1', 'spars5_2','spars5_3',
          'sparsfc6', 'sparsfc7')

effect_size_order = c( spars1_1_r, spars1_2_r, 
                       spars2_1_r, spars2_2_r,
                       spars3_1_r, spars3_2_r, spars3_3_r,
                       spars4_1_r, spars4_2_r, spars4_3_r,
                       spars5_1_r, spars5_2_r, spars5_3_r,
                       sparsfc6_r, sparsfc7_r)

barplot(effect_size_order,  names = xlab, las = 2, col = "yellow" , main = paste('Effect size for ',bdd,' with ',regularization," regularization" ,", sparseness only",sep=""))



