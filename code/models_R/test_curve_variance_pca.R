#####################################################################################
# 1. DESCRIPTION:
#####################################################################################

#####################################################################################
# 2. LIBRAIRIES:
#####################################################################################
library("rjson")
library("readr")
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
library("tidyverse")
library("tibble")
setwd("/home/renoult/Bureau/thesis/code/functions")
#####################################################################################
# 3. Fonctions
#####################################################################################


######################
#3.2 BIG FUNCTION
######################
kfold_gini <- function(bdd, weight, metric, layer, regularization, print_number) {
  
  print('######')
  print(layer)
  
  #chargement du fichier de somme cumulée de variance expliquée par composante
  log_variance =paste('../../results/',bdd,'/pca_variance/', sep="")
  df_variance = read_csv(file = paste(log_variance,"varianceCumule_",layer,".csv", sep =""), show_col_types = TRUE)
  df_variance = df_variance[,-1]
  
  df = t(df_variance)
  
  plot(df)
  
  }
#####################################################################################
# 4. PARAMETERS:
#####################################################################################
bdd <- c('SCUT-FBP')
weight <- c('imagenet')
metric <- c('gini_flatten')
layers <-  c( 'block1_conv1','block1_conv2',
              'block2_conv1','block2_conv2',
              'block3_conv1','block3_conv2','block3_conv3',
              'block4_conv1','block4_conv2','block4_conv3',
              'block5_conv1','block5_conv2','block5_conv3',
              'fc1','fc2')
regularization <- 'lasso' #ridge for ridge, lasso for lasso, glmnet for elasticnet
print_number = 200

set.seed(123)

######################
#5.1 Loop on kfold_gini's function for each layer
######################
for (layer in layers){
  kfold_gini(bdd, weight, metric, layer, regularization, print_number)
}























bdd = 'SCUT-FBP'
layer = 'block1_conv1'
log_path =paste('../../results/',bdd,'/pca/', sep="")
df_pc = read_csv(file = paste(log_path,"pca_values_",layer,".csv", sep =""), show_col_types = FALSE)






