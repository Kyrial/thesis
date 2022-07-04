####################################################################################
# 1. DESCRIPTION:
#####################################################################################

#####################################################################################
# 2. LIBRAIRIES:
#####################################################################################
library("rjson")
library('readr')
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
bdd <- c('SCUT-FBPF')
bdd2 <- c('SCUT-FBPF')
weight <- c('imagenet')
metric <- c('gini_flatten')
subset_db1 = 421 #5000 scut, 827 cfd, 500 MART, 1563 JEN,406 CFDM, 421 CFDF, 109 CFDA, 2750 SCUTF, 4000 SCUTA, 2750 SCUTM
subset_db2 = 421
#####################################################################################
# 4. DATA MANAGEMENT
#####################################################################################

##########################
#4.0 Import des labels
###########################

#SCUT
labels_path = '../../data/redesigned/SCUT-FBP/labels_SCUT-FBP.csv'

labels = read.csv( file = labels_path,sep = ",",stringsAsFactors = TRUE, header  = FALSE )


#####################################################################################
# 4.1 database_1
#####################################################################################

labels_path = paste('../../data/redesigned/',bdd,'/labels_',bdd,'.csv', sep="")
log_path =paste('../../results/',bdd,'/log_', sep="")

#chargement du fichier
df_metrics <- read_csv(file = paste(log_path,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""))

#passage en flottants (avant c'était des strings)
df_metrics = sapply(df_metrics, as.numeric)

#il faut repasser en df après le sapply
df <- as.data.frame(df_metrics)

scaled_df <- scale(df[,-1]) 
df <- cbind(df$rate ,scaled_df) 
df<- as.data.frame(df, optional = TRUE)
df <- plyr::rename(df, c("V1" = "rate"))

#subset
set.seed(173)
df = df[sample(1:nrow(df)), ]
df = df[1:subset_db1,]

#####################################################################################
# 4.2 database_2
#####################################################################################

labels_path2 = paste('../../data/redesigned/',bdd2,'/labels_',bdd2,'.csv', sep="")
log_path2 =paste('../../results/',bdd2,'/log_', sep="")

#chargement du fichier
df_metrics2 <- read_csv(file = paste(log_path2,'_',bdd2,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""))

#passage en flottants (avant c'était des strings)
df_metrics2 = sapply(df_metrics2, as.numeric)

#il faut repasser en df après le sapply
df2 <- as.data.frame(df_metrics2)

scaled_df2 <- scale(df2[,-1]) 
df2 <- cbind(df2$rate ,scaled_df2) #si on avait pas scaled la beauté il aurait fallu la remettre
df2<- as.data.frame(df2, optional = TRUE)
df2 <- plyr::rename(df2, c("V1" = "rate"))

#subset
set.seed(173)
df2 = df2[sample(1:nrow(df2)), ]
df2 = df2[1:subset_db2,]
#####################################################################################
# 5. MODEL: RIDGE REGRESSION
#####################################################################################

matrix = as.matrix(df)
matrix2 = as.matrix(df2)

############
#model
###########

ctrl = trainControl(method = "repeatedcv", number = 10, repeats = 50) #10-fold cv
lambdas = 10^seq(2,-4,by=-0.1)
model = train( rate ~ ., data = df ,method = "glmnet", tuneGrid = expand.grid(alpha = 0, lambda = lambdas),preProc = c("center", "scale"),trControl = ctrl, metric = "Rsquared") #alpha = 0 pour ridge (1 pour lasso)

############
#predictions:
###########
  
#on prédit les notes de beauté du test en fonction du model issu du train
x_test = df2[,-1]
prediction <- model %>% predict(x_test) %>% as.vector()
  
#on fait la corrélation entre les valeurs de beauté prédites et réelles
Rsquare = R2(df2[,1], prediction)
print(Rsquare)
