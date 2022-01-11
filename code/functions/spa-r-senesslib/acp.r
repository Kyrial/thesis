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
library("glmnet")
library("tidyverse")
setwd("/home/renoult/Bureau/internship_cefe_2021/code/functions")
#####################################################################################
# 3. PARAMETERS:
#####################################################################################
model_name <- 'VGG16'
bdd <- c('SMALLTEST')
weight <- c('imagenet')
metric <- c('kurtosis')
#####################################################################################
# 4. DATA MANAGEMENT
#####################################################################################
labels_path ='../../data/redesigned/smalltest/labels_smalltest.csv'
log_path ='../../results/smalltest/log_'

#chargement du fichier
matrix_metrics <- do.call(cbind, fromJSON(file = paste(log_path,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
