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
setwd("/home/renoult/Bureau/thesis/code/functions")
#####################################################################################
# 3. Fonctions
#####################################################################################

######################
#3.1 AIC/BIC
######################
glmnet_cv_aicc <- function(fit, lambda = 'lambda.1se'){
  
  whlm <- which(fit$lambda == fit[[lambda]])
  
  with(fit$glmnet.fit,
       {
         tLL <- nulldev - nulldev * (1 - dev.ratio)[whlm]
         k <- df[whlm]
         n <- nobs
         return(list('AICc' = - tLL + 2 * k + 2 * k * (k + 1) / (n - k - 1),
                     'BIC' = log(n) * k - tLL))
       })
}
######################
#3.2 BIG FUNCTION
######################
kfold_pca <- function(bdd, weight, metric, layer, regularization, print_number) {
  
    print('######')
    print(layer)
    ######################
    # 3.2.1 DATA MANAGEMENT
    ######################
    labels_path = paste('../../data/redesigned/',bdd,'/labels_',bdd,'.csv', sep="")
    log_path =paste('../../results/',bdd,'/pca/', sep="")
    log_path_rate =paste('../../results/',bdd,'/log_', sep="")
    
    #chargement du fichier
    df_pc = read_csv(file = paste(log_path,"pca_values_",layer,".csv", sep =""), show_col_types = FALSE)
    df_pc = df_pc[,-1]
    
    #on récupère les notes de beauté
    matrix_metrics <- do.call(cbind, fromJSON(file = paste(log_path_rate,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
    df_metrics <- as.data.frame(matrix_metrics, optional = TRUE)
    df_metrics = sapply(df_metrics, as.numeric)
    df_metrics <- as.data.frame(df_metrics)
    df = cbind(df_metrics$rate, df_pc)
    df <- plyr::rename(df, c("df_metrics$rate" = "rate"))
    
    ###############################
    # 3.2.2. MODEL: REGRESSION WITH REGULARIZATION (ridge, lasso or elasticnet)
    ###############################
    ctrl = trainControl(method = "repeatedcv", number = 10) #10-fold cv
    model1 = train( rate ~ ., data = df ,method = regularization,preProc = c("center", "scale"),trControl = ctrl, metric = "Rsquared")
    
    alpha = model1$results$alpha[1]
    lambda = model1$results$lambda[1]
    r_squared = model1$results$Rsquared[1]
    
    matrix = as.matrix(df)
    x = matrix[,-1]
    y = matrix[,1]
    if (regularization == 'glmnet'){
      model2 = cv.glmnet(x, y, alpha = alpha)
    } else if (regularization == 'lasso'){
      model2 = cv.glmnet(x, y, alpha = 1)
    } else { #cad regularization = ridge
      model2 = cv.glmnet(x, y, alpha = 1)
    }
    
    criterions =  glmnet_cv_aicc(model2)
    
    list = list('r_squared' = r_squared, 'AIC'= criterions$AICc, 'BIC'= criterions$BIC )
    
    return(list)

}
#####################################################################################
# 4. PARAMETERS:
#####################################################################################
bdd <- c('JEN')
weight <- c('imagenet')
metric <- c('gini_flatten')
layers <-   c('input_1',
            'block1_conv1','block1_conv2','block1_pool',
            'block2_conv1','block2_conv2','block2_pool',
            'block3_conv1','block3_conv2','block3_conv3','block3_pool',
            'block4_conv1','block4_conv2','block4_conv3','block4_pool',
            'block5_conv1','block5_conv2','block5_conv3','block5_pool',
            'fc1','fc2',
            'flatten')
regularization <- 'lasso' #0 for ridge, 1 for lasso
print_number = 200

set.seed(123)

######################################################################################
# 5. MAIN:
######################################################################################
R_squareds = c()
AICs = c()
BICs = c()

for (layer in layers){
  results = kfold_pca(bdd, weight, metric, layer, regularization, print_number)
  R_squareds = c(R_squareds, results$r_squared)
  AICs = c(AICs, results$AIC)
  BICs = c(BICs, results$BIC)
}

print('## R2 ##')
print(R_squareds)
print('## AICs ##')
print(AICs)
print('## BICs ##')
print(BICs)

barplot(R_squareds, names.arg = layers, xlab = "layers", ylab= "rsquared", main = cbind('R2_',regularization))
barplot(AICs, names.arg = layers, xlab = "layers", ylab= "rsquared", main = cbind('AIC_',regularization))
barplot(BICs, names.arg = layers, xlab = "layers", ylab= "rsquared", main = cbind('BIC_',regularization))



