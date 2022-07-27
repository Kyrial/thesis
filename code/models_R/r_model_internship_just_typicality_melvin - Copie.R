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

library(caret)
library(glmnet)
setwd("/home/renoult/Bureau/thesis/code/functions")
setwd("/Users/koala/source/repos/thesis/code/functions")
#####################################################################################
# 3. PARAMETRES: def analyse_metrics(model_name, bdd, weight, metric,k):
#####################################################################################
#####################################################################################
# 3.1 Parametres
#####################################################################################
#mettre ça pas en dur a terme mais en paramètres passé au script python (ou pas?)

model_name <- 'VGG16'
bdd <- 'CFD_WM'
bdd <- 'CFD_AF'
bdd <- 'MART'
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
                       #'flatten',
                     'fc1', 'fc2'
                     )
          
         
        #path d'enregistrement des résultats et chargement des données  
        
        labels_path = paste('../../data/redesigned/',bdd,'/labels_',bdd,'.csv', sep='')
        #log_path =paste('../../results/',bdd,'/LLH_FeatureMap/LLH_',bdd,'_AllLLH.csv',sep = '')
        #log_path =paste('../../results/',bdd,'/LLH_max/LLH_',bdd,'_AllLLH.csv',sep = '')
        log_path =paste('../../results/',bdd,'/LLH_average/LLH_',bdd,'_AllLLH.csv',sep = '')
        #log_path =paste('../../results/',bdd,'/LLH_average_model/LLH_',bdd,'_AllLLH.csv',sep = '')
        log_path =paste('../../results/',bdd,'/LLH/LLH_',bdd,'_AllLLH.csv',sep = '')
        #log_path =paste('../../results/',bdd,'/LLH_pca/LLH_',bdd,'_AllLLH.csv',sep = '')
        log_path_rate =paste('../../results/',bdd,'/log_', sep="")
        
        
        matrix_metrics <- read_csv(file=log_path)
        colnames(matrix_metrics)[2] <- 'input_1'
        
       
        matrix_beauty <- do.call(cbind,read.csv(file=labels_path, header=FALSE))
        matrix_beauty <- matrix_beauty[-c(1),]
        colnames(matrix_beauty) <- c("img","rate")
        df_beauty <-subset(matrix_beauty, select = c(rate))
    # df_beauty$rate <-as.numeric(df_beauty$rate)
        
        #on récupère les notes de beauté
      #  matrix_beauty <- do.call(cbind, fromJSON(file = paste(log_path_rate,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
      #  df_beauty <- as.data.frame(matrix_beauty, optional = TRUE)
      #  df_beauty = sapply(df_beauty, as.numeric)
      #  df_beauty <- as.data.frame(df_beauty)
      #  df = cbind(df_beauty$rate, matrix_metrics)
        
        df = cbind(df_beauty, matrix_metrics)
        #df <- plyr::rename(df, c("df_beauty$rate" = "rate"))
        df <- df[,-2]
        
        
        df_metrics = sapply(df, as.numeric)
        df_metrics <- as.data.frame(df_metrics)


        #################
        # 5. MULTIPLES MODELS
        #####################################################################################
        
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
                                    #      'flatten' = 'flatten',
                                          'fc6_relu' = 'fc1',
                                          'fc7_relu' = 'fc2'
                                         ))
          
        }
        print(paste('parameters are:',bdd,'-',weight,'-',metric, sep = ""))

        
        
      
        
        
        #####################################################################################
        #5.3. model with layers and interaction with complexity
        #####################################################################################
        #####################################################################################
        # MODELE LINEAIRE
        #####################################################################################
        model = lm(rate ~ +conv1_1+conv1_2+conv2_1+conv2_2+conv3_1+conv3_2+conv3_3+conv4_1+conv4_2+conv4_3+conv5_1+conv5_2+conv5_3
                   +fc6_relu+fc7_relu
                   ,data = df_metrics)
        #ajouter les couches dense
       print(summary(model))
      
       
       
plot(model)
       
       
      
       
       
       ##enlever les layer en trop et mettre rate en 1
       df_metrics <-subset(df_metrics, select = c(rate, conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3,conv4_1,conv4_2,conv4_3,conv5_1,conv5_2,conv5_3
                                                 ,fc6_relu, fc7_relu))
    
     #  df_metrics <- cbind((df_beauty),(df_metrics))
    #   df_metrics <- as.data.frame(df_metrics)
       #######Nico Ridge 2
       k= nrow(df_metrics)
       print(k)
       matrix = as.matrix(df_metrics)
       
       lambdas = c()
       predictions = c()
       lambdaMin = c()
       for (j in 1:100){
        cv_train <- cv.glmnet(matrix[,-1], matrix[,1], alpha = 0) #alpha = 0 fait une ridge regression (1 si lasso)
        #print(cv_train$lambda.min)
        #plot(cv_train, xvar='lambda')
        lambdaMin <-c(lambdaMin,cv_train$lambda.min)
       }
       hist(lambdaMin,20)
       la <- median(lambdaMin)
       
       for (i in 1:k){
         
         print(i)
         
         train = matrix[-i,]
        test = matrix[i,]
         
         x_train = train[,-1] ##df_beauty[-i,]
         y_train = train[,1]
         
         #print(i)
     #    cv_train <- cv.glmnet(x_train, y_train, alpha = 0) #alpha = 0 fait une ridge regression (1 si lasso)
    #     print(cv_train$lambda.min)
    #     plot(cv_train, xvar='lambda')
         
         model <- glmnet(x_train, y_train, alpha = 0 , lambda = la ) #cv_train$lambda.min)
         #model <- lm( y_train~x_train)
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
       
       matrix2 <- cbind(predictions ,matrix)
       Rsquare = R2(matrix2[,1], matrix2[,2])
       print(Rsquare)
       
       
       
       
       
       
       ###########################################################
       ###########################################################
       ###########################################################
       
       
       
       
       index = sample(1:nrow(df_metrics), 0.7*nrow(df_metrics)) 
       
       train = df_metrics[index,] # Create the training data 
       test = df_metrics[-index,] # Create the test data
       
       cols_reg = c("rate", "conv1_1","conv1_2","conv2_1","conv2_2","conv3_1","conv3_2","conv3_3","conv4_1","conv4_2","conv4_3","conv5_1","conv5_2","conv5_3"
                  ,"fc6_relu","fc7_relu"
                    )
       
       dummies <- dummyVars(rate ~ ., data = df_metrics[,cols_reg])
       
       train_dummies = predict(dummies, newdata = train[,cols_reg])
       
       test_dummies = predict(dummies, newdata = test[,cols_reg])
       
       print(dim(train_dummies)); print(dim(test_dummies))

       x = as.matrix(train_dummies)
       y_train = train$rate
       
       x_test = as.matrix(test_dummies)
       y_test = test$rate
       
       lambdas <- 10^seq(2, -3, by = -.1)
       ridge_reg = glmnet(x, y_train, nlambda = 25, alpha = 0, family = 'gaussian', lambda = lambdas)
       
       print(summary(ridge_reg))
       cv_ridge <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
       
       
       
       optimal_lambda <- cv_ridge$lambda.min
       # Compute R^2 from true and predicted values
       eval_results <- function(true, predicted, df) {
         SSE <- sum((predicted - true)^2)
         SST <- sum((true - mean(true))^2)
         R_square <- 1 - SSE / SST
         RMSE = sqrt(SSE/nrow(df))
         
         
         # Model performance metrics
         data.frame(
           RMSE = RMSE,
           Rsquare = R_square
         )
         
       }
       #####################################################################################
       #5.4. REGRESSION MULTIPLE REGULARISER RIDGE
       #####################################################################################
       
       # Prediction and evaluation on train data
       predictions_train <- predict(ridge_reg, s = optimal_lambda, newx = x)
       eval_results(y_train, predictions_train, train)
       
       # Prediction and evaluation on test data
       predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
       eval_results(y_test, predictions_test, test)
       
       
       
       
       
       
       
       
       
       
       #####################################################################################
       #5.4. LASSO
       #####################################################################################
       
       
       
       
       lambdas <- 10^seq(2, -3, by = -.1)
       
       # Setting alpha = 1 implements lasso regression
       lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)
       
       # Best 
       lambda_best <- lasso_reg$lambda.min 
       lambda_best
       
       lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)
       
       predictions_train <- predict(lasso_model, s = lambda_best, newx = x)
       eval_results(y_train, predictions_train, train)
       
       predictions_test <- predict(lasso_model, s = lambda_best, newx = x_test)
       eval_results(y_test, predictions_test, test)
       
       
       
       #####################################################################################
       #5.4. ELASTIC NET
       #####################################################################################
       
       
       # Set training control
       train_cont <- trainControl(method = "repeatedcv",
                                  number = 10,
                                  repeats = 5,
                                  search = "random",
                                  verboseIter = FALSE)
       
       # Train the model
       elastic_reg <- train(rate ~ .,
                            data = train[,cols_reg],
                            method = "glmnet",
                            preProcess = c("center", "scale"),
                            tuneLength = 10,
                            trControl = train_cont)
       
       
       # Best tuning parameter
       elastic_reg$bestTune
       
       # Make predictions on training set
       predictions_train <- predict(elastic_reg, x)
       eval_results(y_train, predictions_train, train) 
       
       # Make predictions on test set
       predictions_test <- predict(elastic_reg, x_test)
       eval_results(y_test, predictions_test, test)
       
       
      