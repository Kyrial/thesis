####################################################################################
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
library("glmnet")
setwd("/home/renoult/Bureau/thesis/code/functions")
#####################################################################################
# 3. PARAMETERS:
#####################################################################################
model_name <- 'VGG16'
bdd <- c('CFD')
bdd2 <- c('SCUT-FBP')
bdd3 <- c('MART')
weight <- c('imagenet')
metric <- c('gini_flatten')
#####################################################################################
# 4. DATA MANAGEMENT
#####################################################################################

#####################################################################################
# 4.1 database_1
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


#création d'un dataframe avec la complexité, la sparsité approximée par gini
df <- cbind(df_metrics, df_complexity)

#Z-transformation (centré réduit)
scaled_df <- scale(df[,-1]) #df[,-1] pour ne pas z transformer la beauté
df <- cbind(df$rate ,scaled_df) #si on avait pas scaled la beauté il aurait fallu la remettre
#df = scaled_df
df<- as.data.frame(df, optional = TRUE)
df <- plyr::rename(df, c("V1" = "rate"))

#####################################################################################
# 4.2 database_2
#####################################################################################

labels_path2 = paste('../../data/redesigned/',bdd2,'/labels_',bdd2,'.csv', sep="")
log_path2 =paste('../../results/',bdd2,'/log_', sep="")

#chargement du fichier
matrix_metrics2 <- do.call(cbind, fromJSON(file = paste(log_path2,'_',bdd2,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))

#si on ne fait pas ça, l'input peut avoir un indice variable
colnames(matrix_metrics2)[2] <- 'input_1'

#idem avec les calculs de complexité
matrix_complexity2 <- do.call(cbind, fromJSON(file = paste(log_path2,'_',bdd2,'_',weight,'_','mean','_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
colnames(matrix_complexity2)[2] <- 'input_1'

#passage des matrice en dataframe
df_metrics2 <- as.data.frame(matrix_metrics2, optional = TRUE)
df_complexity2 <- as.data.frame(matrix_complexity2, optional = TRUE)        
#passage en flottants (avant c'était des strings)
df_metrics2 = sapply(df_metrics2, as.numeric)
df_complexity2 = sapply(df_complexity2, as.numeric)

#il faut repasser en df après le sapply
df_metrics2 <- as.data.frame(df_metrics2)
df_complexity2 <- as.data.frame(df_complexity2[,-1])

#changement des noms de colonne pour les uniformiser car les differents weights ont des noms de layers différents
df_metrics2 = plyr::rename(df_metrics2, c("input_1" = "input_1",
                                        'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                        'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                        'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                        'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                        'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                        'flatten'='flatten','fc1'='fc6_relu','fc2'='fc7_relu'))
#même démarche pour la complexité
df_complexity2 = plyr::rename(df_complexity2, c("input_1" = "input_1_comp",
                                              'block1_conv1'='conv1_1_comp','block1_conv2'='conv1_2_comp','block1_pool'='pool1_comp',
                                              'block2_conv1'='conv2_1_comp','block2_conv2'='conv2_2_comp','block2_pool'='pool2_comp',
                                              'block3_conv1'='conv3_1_comp','block3_conv2'='conv3_2_comp','block3_conv3'='conv3_3_comp','block3_pool'='pool3_comp',
                                              'block4_conv1'='conv4_1_comp','block4_conv2'='conv4_2_comp','block4_conv3'='conv4_3_comp','block4_pool'='pool4_comp',
                                              'block5_conv1'='conv5_1_comp','block5_conv2'='conv5_2_comp','block5_conv3'='conv5_3_comp','block5_pool'='pool5_comp',
                                              'flatten'='flatten_comp','fc1'='fc6_relu_comp','fc2'='fc7_relu_comp'))


#création d'un dataframe avec la complexité, la sparsité approximée par gini
df2 <- cbind(df_metrics2, df_complexity2)

#Z-transformation (centré réduit)
scaled_df2 <- scale(df2[,-1]) #df[,-1] pour ne pas z transformer la beauté
df2 <- cbind(df2$rate ,scaled_df2) #si on avait pas scaled la beauté il aurait fallu la remettre
#df = scaled_df
df2<- as.data.frame(df2, optional = TRUE)
df2 <- plyr::rename(df2, c("V1" = "rate"))


#####################################################################################
# 4.3 database_3
#####################################################################################

labels_path3 = paste('../../data/redesigned/',bdd3,'/labels_',bdd3,'.csv', sep="")
log_path3 =paste('../../results/',bdd3,'/log_', sep="")

#chargement du fichier
matrix_metrics3 <- do.call(cbind, fromJSON(file = paste(log_path3,'_',bdd3,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))

#si on ne fait pas ça, l'input peut avoir un indice variable
colnames(matrix_metrics3)[2] <- 'input_1'

#idem avec les calculs de complexité
matrix_complexity3 <- do.call(cbind, fromJSON(file = paste(log_path3,'_',bdd3,'_',weight,'_','mean','_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
colnames(matrix_complexity3)[2] <- 'input_1'

#passage des matrice en dataframe
df_metrics3 <- as.data.frame(matrix_metrics3, optional = TRUE)
df_complexity3 <- as.data.frame(matrix_complexity3, optional = TRUE)        
#passage en flottants (avant c'était des strings)
df_metrics3 = sapply(df_metrics3, as.numeric)
df_complexity3 = sapply(df_complexity3, as.numeric)

#il faut repasser en df après le sapply
df_metrics3 <- as.data.frame(df_metrics3)
df_complexity3 <- as.data.frame(df_complexity3[,-1])

#changement des noms de colonne pour les uniformiser car les differents weights ont des noms de layers différents
df_metrics3 = plyr::rename(df_metrics3, c("input_1" = "input_1",
                                          'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                          'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                          'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                          'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                          'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                          'flatten'='flatten','fc1'='fc6_relu','fc2'='fc7_relu'))
#même démarche pour la complexité
df_complexity3 = plyr::rename(df_complexity3, c("input_1" = "input_1_comp",
                                                'block1_conv1'='conv1_1_comp','block1_conv2'='conv1_2_comp','block1_pool'='pool1_comp',
                                                'block2_conv1'='conv2_1_comp','block2_conv2'='conv2_2_comp','block2_pool'='pool2_comp',
                                                'block3_conv1'='conv3_1_comp','block3_conv2'='conv3_2_comp','block3_conv3'='conv3_3_comp','block3_pool'='pool3_comp',
                                                'block4_conv1'='conv4_1_comp','block4_conv2'='conv4_2_comp','block4_conv3'='conv4_3_comp','block4_pool'='pool4_comp',
                                                'block5_conv1'='conv5_1_comp','block5_conv2'='conv5_2_comp','block5_conv3'='conv5_3_comp','block5_pool'='pool5_comp',
                                                'flatten'='flatten_comp','fc1'='fc6_relu_comp','fc2'='fc7_relu_comp'))


#création d'un dataframe avec la complexité, la sparsité approximée par gini
df3 <- cbind(df_metrics3, df_complexity3)

#Z-transformation (centré réduit)
scaled_df3 <- scale(df3[,-1]) #df[,-1] pour ne pas z transformer la beauté
df3 <- cbind(df3$rate ,scaled_df3) #si on avait pas scaled la beauté il aurait fallu la remettre
#df = scaled_df
df3<- as.data.frame(df3, optional = TRUE)
df3 <- plyr::rename(df3, c("V1" = "rate"))



#####################################################################################
# 5. MODEL: RIDGE REGRESSION
#####################################################################################
set.seed(123)

matrix = as.matrix(df)
matrix2 = as.matrix(df2)
matrix3 = as.matrix(df3)

#on définit le train et le test
train = rbind(matrix,matrix2)
test = matrix3

#on définit les variables explicatives (mesures de fluence et de complexité) et la variable a expliquer (note de beauté)
x_train = train[,-1]
y_train = train[,1]
  
############
#ridge/lasso
###########

#on fait un premier train pour estimer le lambda
cv_train <- cv.glmnet(x_train, y_train, alpha = 1) #alpha = 0 fait une ridge regression (1 si lasso)
  
#on train le vrai model avec le lambda estimé
model <- glmnet(x_train, y_train, alpha = 1, lambda = cv_train$lambda.min) #alpha = 0 fait une ridge regression (1 si lasso)
  
############
#elastic net (ne marche pas pour le moment)
###########
  
#model <- train(
#  rate ~., data = train, method = "glmnet",
#  trControl = trainControl("cv", number = 10),
#  tuneLength = 10
#)
  
############
#predictions:
###########
  
#on prédit les notes de beauté du test en fonction du model issu du train
x_test = test[,-1]
prediction <- model %>% predict(x_test) %>% as.vector()
  
#on fait la corrélation entre les valeurs de beauté prédites et réelles
Rsquare = R2(matrix3[,1], prediction)
print(Rsquare)
