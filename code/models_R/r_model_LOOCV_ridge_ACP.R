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
setwd("/home/renoult/Bureau/thesis/code/functions")
#####################################################################################
# 3. Fonction
#####################################################################################

loocv_pca <- function(bdd, weight, metric, layer, regularization, print_number) {
  
  print('######')
  print(layer)
  ######################
  # 3.1 DATA MANAGEMENT
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
  # 3.2. MODEL: RIDGE REGRESSION
  ###############################
  
  variables = colnames(df[,-1])
  
  matrix = as.matrix(df)
  
  k = nrow(matrix)
  
  lambdas = c()
  predictions = c()
  
  for (i in 1:k){
    
    if (i%%print_number == 0){
      print(i)
    }
    
    train = matrix[-i,]
    test = matrix[i,]
    
    x_train = train[,-1]
    y_train = train[,1]
    
    cv_train <- cv.glmnet(x_train, y_train, alpha = regularization) #alpha = 0 fait une ridge regression (1 si lasso)
    
    model <- glmnet(x_train, y_train, alpha = regularization, lambda = cv_train$lambda.min)
    
    lambdas = c(lambdas, cv_train$lambda.min)
    
    #elastic net
    
    #model <- train(
    #  rate ~., data = train, method = "glmnet",
    #  trControl = trainControl("cv", number = 10),
    #  tuneLength = 10
    #)
    
    
    
    #predictions:
    x_test = test[-1]
    prediction <- model %>% predict(x_test) %>% as.vector()
    
    predictions = c(predictions, prediction)
    
    
  } 
  
  matrix <- cbind(predictions ,matrix)
  Rsquare = R2(matrix[,1], matrix[,2])
  return(Rsquare)
}
#####################################################################################
# 4. PARAMETERS:
#####################################################################################
bdd <- c('MART')
weight <- c('imagenet')
metric <- c('gini_flatten')
layers <- c('input_1',
            'block1_conv1','block1_conv2','block1_pool',
            'block2_conv1','block2_conv2','block2_pool',
            'block3_conv1','block3_conv2','block3_conv3','block3_pool',
            'block4_conv1','block4_conv2','block4_conv3','block4_pool',
            'block5_conv1','block5_conv2','block5_conv3','block5_pool',
            'fc1','fc2',
            'flatten')
regularization <- 1 #0 ridge, 1 lasso
print_number = 200

set.seed(123)

r_squareds = c()

for (layer in layers){
  r_squared = loocv_pca(bdd, weight, metric, layer, regularization, print_number)
  r_squareds = c(r_squareds, r_squared)
}

print(r_squareds)
barplot(r_squareds, names.arg = layers, xlab = "layers", ylab= "rsquared")



