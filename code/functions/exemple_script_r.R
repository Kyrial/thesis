#DIBOT Nicolas
#TP B
#M1 Bioinformatique-Biostatistique, option Bioinformatique

#12/02/2020

#version de R utilisée: 3.6.2





############################
####ETAPES PRELIMINAIRES####################################################################################
############################

## Creation d'une fonction automatisant le calcul de l'IC95%
confEstimate <- function (colonne) {
  
  n <- sum(!is.na(colonne))  #recodage de la fonction par rapport à celle donnée pour qu'elle ne prenne qu'un argument
  et <- sd(colonne,na.rm=TRUE)
  moy <- mean(colonne,na.rm=TRUE)
  inf <- moy - (1.96*(et/sqrt(n)))
  sup <- moy + (1.96*(et/sqrt(n)))
  cat("IC95%: [",inf,";",sup,"]", "\n") }
  
#importations
library(readr)

setwd("C:/RTravail/tp_DM")

donnees <- read.csv2("Imputation.csv",sep=";", dec=".")

#génération jeu de données
set.seed(4)
donnees$pas.M6 <- sample(donnees$pas.M6)

#création d'une colonne delta
donnees$deltaPAS = donnees$pas.M6 - donnees$pas.M0





############################
#3.1: DONNEES INITIALES#####################################################################################
############################

#création de deux jeux de données en fonction du groupe

gp1 <- subset(donnees, donnees$groupe=="Trt1")
gp2 <- subset(donnees, donnees$groupe=="Trt2")

# PAS M0 groupe1
mean(gp1$pas.M0,na.rm=TRUE)
sum(!is.na(gp1$pas.M0))
confEstimate(gp1$pas.M0)

# PAS M0 groupe2
mean(gp2$pas.M0,na.rm=TRUE)
sum(!is.na(gp2$pas.M0))
confEstimate(gp2$pas.M0)

# PAS M6 groupe 1
mean(gp1$pas.M6,na.rm=TRUE)
sum(!is.na(gp1$pas.M6))
confEstimate(gp1$pas.M6)

# PAS M6 groupe 2
sum(!is.na(gp2$pas.M6))
mean(gp2$pas.M6,na.rm=TRUE)
confEstimate(gp2$pas.M6)






###############################
#3.2: COMPLETE CASE ANALYSIS###################################################################################
###############################


donnees.complete.pas.M6<- subset(donnees,  !is.na(donnees$pas.M6)) #subset sans NA dans M6


donnees.trt1.pas.M6 <- subset(donnees.complete.pas.M6, donnees.complete.pas.M6$groupe=="Trt1") #subset sans NA dans M6 du groupe TRT1 
donnees.trt2.pas.M6 <- subset(donnees.complete.pas.M6, donnees.complete.pas.M6$groupe=="Trt2") #subset sans NA dans M6 du groupe TRT2


# PAS M0 groupe1
mean(donnees.trt1.pas.M6$pas.M0,na.rm=TRUE)
sum(!is.na(donnees.trt1.pas.M6$pas.M0))
confEstimate(donnees.trt1.pas.M6$pas.M0)

# PAS M0 groupe2
mean(donnees.trt2.pas.M6$pas.M0,na.rm=TRUE)
sum(!is.na(donnees.trt2.pas.M6$pas.M0))
confEstimate(donnees.trt2.pas.M6$pas.M0)

# PAS M6 groupe 1
mean(donnees.trt1.pas.M6$pas.M6,na.rm=TRUE)
sum(!is.na(donnees.trt1.pas.M6$pas.M6))
confEstimate(donnees.trt1.pas.M6$pas.M6)

# PAS M6 groupe 2
sum(!is.na(donnees.trt2.pas.M6$pas.M6))
mean(donnees.trt2.pas.M6$pas.M6,na.rm=TRUE)
confEstimate(donnees.trt2.pas.M6$pas.M6)

# PAS delta groupe1
mean(donnees.trt1.pas.M6$deltaPAS,na.rm=TRUE)
sum(!is.na(donnees.trt1.pas.M6$deltaPAS))
confEstimate(donnees.trt1.pas.M6$deltaPAS)

# PAS delta groupe2
mean(donnees.trt2.pas.M6$deltaPAS,na.rm=TRUE)
sum(!is.na(donnees.trt2.pas.M6$deltaPAS))
confEstimate(donnees.trt2.pas.M6$deltaPAS)

# PAS delta Trt1-Trt2	
mean(donnees.trt1.pas.M6$deltaPAS,na.rm=TRUE) - mean(donnees.trt2.pas.M6$deltaPAS,na.rm=TRUE)


#comparaison des delta entre M0 et M6  pour chaque groupe
t.test(donnees.trt1.pas.M6$deltaPAS, donnees.trt2.pas.M6$deltaPAS)  
#on récupère l'IC du ttest ici






#############################
#3.3: Worst and best cases###################################################################################
#############################

donnees.exo<-donnees
donnees.exo$pas.M6.worst<-pmax(donnees.exo$pas.M0,donnees.exo$pas.M3,na.rm=TRUE)
#on cherche a faire BAISSER la pression (tendance à la baisse entre M0 et M6 
#donc une pression haute est le worst case)
donnees.exo$pas.M6.best<-pmin(donnees.exo$pas.M0,donnees.exo$pas.M3,na.rm=TRUE) 
#on cherche a faire BAISSER la pression (tendance à la baisse entre M0 et M6 
#donc une pression basse est le best case)


#on recode, si il y a une NA en PASM6, on remplace par bestcase (ou worstcase), sinon on garde PASM6
donnees.exo$pas.M6.best <- ifelse(is.na(donnees.exo$pas.M6),donnees.exo$pas.M6.best,donnees.exo$pas.M6)
donnees.exo$pas.M6.worst <- ifelse(is.na(donnees.exo$pas.M6),donnees.exo$pas.M6.worst,donnees.exo$pas.M6)

donnees.exo$deltaPASbest <- donnees.exo$pas.M6.best - donnees.exo$pas.M0
donnees.exo$deltaPASworst <- donnees.exo$pas.M6.worst - donnees.exo$pas.M0


gp1.exo <- subset(donnees.exo, donnees.exo$groupe=="Trt1")  #table spécifique au groupe 1
gp2.exo <- subset(donnees.exo, donnees.exo$groupe=="Trt2")  #table spécifique au groupe 2


############
##BESTCASE##
############

# PAS M0 groupe 1
mean(gp1.exo$pas.M0,na.rm=TRUE)
sum(!is.na(gp1.exo$pas.M0))
confEstimate(gp1.exo$pas.M0)

# PAS M0 groupe2
mean(gp2.exo$pas.M0,na.rm=TRUE)
sum(!is.na(gp2.exo$pas.M0))
confEstimate(gp2.exo$pas.M0)

# PAS M6 groupe 1
mean(gp1.exo$pas.M6.best,na.rm=TRUE)
sum(!is.na(gp1.exo$pas.M6.best))
confEstimate(gp1.exo$pas.M6.best)

# PAS M6 groupe 2
sum(!is.na(gp2.exo$pas.M6.best))
mean(gp2.exo$pas.M6.best,na.rm=TRUE)
confEstimate(gp2.exo$pas.M6.best)

# PAS delta groupe1
mean(gp1.exo$deltaPASbest,na.rm=TRUE)
sum(!is.na(gp1.exo$deltaPASbest))
confEstimate(gp1.exo$deltaPASbest)

# PAS delta groupe2
mean(gp2.exo$deltaPASbest,na.rm=TRUE)
sum(!is.na(gp2.exo$deltaPASbest))
confEstimate(gp2.exo$deltaPASbest)

# PAS delta de toutes les données
mean(gp1.exo$deltaPASbest,na.rm=TRUE) - mean(gp2.exo$deltaPASbest,na.rm=TRUE)

#comparaison des delta entre M0 et M6  pour chaque groupe
t.test(gp1.exo$deltaPASbest, gp2.exo$deltaPASbest)  


############
##WORSTCASE#
############

# PAS M0 groupe 1
mean(gp1.exo$pas.M0,na.rm=TRUE)
sum(!is.na(gp1.exo$pas.M0))
confEstimate(gp1.exo$pas.M0)

# PAS M0 groupe2
mean(gp2.exo$pas.M0,na.rm=TRUE)
sum(!is.na(gp2.exo$pas.M0))
confEstimate(gp2.exo$pas.M0)

# PAS M6 groupe 1
mean(gp1.exo$pas.M6.worst,na.rm=TRUE)
sum(!is.na(gp1.exo$pas.M6.worst))
confEstimate(gp1.exo$pas.M6.worst)

# PAS M6 groupe 2
sum(!is.na(gp2.exo$pas.M6.worst))
mean(gp2.exo$pas.M6.worst,na.rm=TRUE)
confEstimate(gp2.exo$pas.M6.worst)

# PAS delta groupe1
mean(gp1.exo$deltaPASworst,na.rm=TRUE)
sum(!is.na(gp1.exo$deltaPASworst))
confEstimate(gp1.exo$deltaPASworst)

# PAS delta groupe2
mean(gp2.exo$deltaPASworst,na.rm=TRUE)
sum(!is.na(gp2.exo$deltaPASworst))
confEstimate(gp2.exo$deltaPASworst)

# PAS delta des deltas de chaque groupe
mean(gp1.exo$deltaPASworst,na.rm=TRUE) - mean(gp2.exo$deltaPASworst,na.rm=TRUE)


#comparaison des delta entre M0 et M6  pour chaque groupe
t.test(gp1.exo$deltaPASworst, gp2.exo$deltaPASworst)  






#############################
#3.4: Personal Mean Score#################################################################################"
#############################

donnees.exo.PMS <- donnees.exo
donnees.exo.PMS$pas.M6.meansubj <- ifelse(is.na(donnees.exo.PMS$pas.M6),(donnees.exo.PMS$pas.M0 + donnees.exo.PMS$pas.M3)/2,donnees.exo.PMS$pas.M6)
donnees.exo.PMS$deltaPAS.PMS <- donnees.exo.PMS$pas.M6.meansubj - donnees.exo.PMS$pas.M0

gp1.exo.PMS <- subset(donnees.exo.PMS, donnees.exo.PMS$groupe=="Trt1")  #table spécifique au groupe 1
gp2.exo.PMS <- subset(donnees.exo.PMS, donnees.exo.PMS$groupe=="Trt2")  #table spécifique au groupe 2

# PAS M0 groupe 1
mean(gp1.exo.PMS$pas.M0,na.rm=TRUE)
sum(!is.na(gp1.exo.PMS$pas.M0))
confEstimate(gp1.exo.PMS$pas.M0)

# PAS M0 groupe2
mean(gp2.exo.PMS$pas.M0,na.rm=TRUE)
sum(!is.na(gp2.exo.PMS$pas.M0))
confEstimate(gp2.exo.PMS$pas.M0)

# PAS M6 groupe 1
mean(gp1.exo.PMS$pas.M6.meansubj,na.rm=TRUE)
sum(!is.na(gp1.exo.PMS$pas.M6.meansubj))
confEstimate(gp1.exo.PMS$pas.M6.meansubj)

# PAS M6 groupe 2
sum(!is.na(gp2.exo.PMS$pas.M6.meansubj))
mean(gp2.exo.PMS$pas.M6.meansubj,na.rm=TRUE)
confEstimate(gp2.exo.PMS$pas.M6.meansubj)

# PAS delta groupe1
mean(gp1.exo.PMS$deltaPAS.PMS,na.rm=TRUE)
sum(!is.na(gp1.exo.PMS$deltaPAS.PMS))
confEstimate(gp1.exo.PMS$deltaPAS.PMS)
#ici on a que 164 sujets car il y a un NA en M0 pour un individu

# PAS delta groupe2
mean(gp2.exo.PMS$deltaPAS.PMS,na.rm=TRUE)
sum(!is.na(gp2.exo.PMS$deltaPAS.PMS))
confEstimate(gp2.exo.PMS$deltaPAS.PMS)

# PAS delta de toutes les données
mean(gp1.exo.PMS$deltaPAS.PMS,na.rm=TRUE) - mean(gp2.exo.PMS$deltaPAS.PMS,na.rm=TRUE)

#comparaison des delta entre M0 et M6  pour chaque groupe
t.test(gp1.exo.PMS$deltaPAS.PMS, gp2.exo.PMS$deltaPAS.PMS)  







####################
#3.5: Mean Score####################################################################################
####################

donnees.exo$pas.M6.meanpop <- ifelse(is.na(donnees.exo$pas.M6),mean(donnees.exo$pas.M6,na.rm=TRUE),donnees.exo$pas.M6)
donnees.exo$deltaPAS.MS <- donnees.exo$pas.M6.meanpop - donnees.exo$pas.M0

gp1.exo.MS <- subset(donnees.exo, donnees.exo$groupe=="Trt1")  #table spécifique au groupe 1
gp2.exo.MS <- subset(donnees.exo, donnees.exo$groupe=="Trt2")  #table spécifique au groupe 2

# PAS M0 groupe 1
mean(gp1.exo.MS$pas.M0,na.rm=TRUE)
sum(!is.na(gp1.exo.MS$pas.M0))
confEstimate(gp1.exo.MS$pas.M0)

# PAS M0 groupe2
mean(gp2.exo.MS$pas.M0,na.rm=TRUE)
sum(!is.na(gp2.exo.MS$pas.M0))
confEstimate(gp2.exo.MS$pas.M0)

# PAS M6 groupe 1
mean(gp1.exo.MS$pas.M6.meanpop,na.rm=TRUE)
sum(!is.na(gp1.exo.MS$pas.M6.meanpop))
confEstimate(gp1.exo.MS$pas.M6.meanpop)

# PAS M6 groupe 2
sum(!is.na(gp2.exo.MS$pas.M6.meanpop))
mean(gp2.exo.MS$pas.M6.meanpop,na.rm=TRUE)
confEstimate(gp2.exo.MS$pas.M6.meanpop)

# PAS delta groupe1
mean(gp1.exo.MS$deltaPAS.MS,na.rm=TRUE)
sum(!is.na(gp1.exo.MS$deltaPAS.MS))
confEstimate(gp1.exo.MS$deltaPAS.MS)

# PAS delta groupe2
mean(gp2.exo.MS$deltaPAS.MS,na.rm=TRUE)
sum(!is.na(gp2.exo.MS$deltaPAS.MS))
confEstimate(gp2.exo.MS$deltaPAS.MS)

# PAS delta de toutes les données
mean(gp1.exo.MS$deltaPAS.MS,na.rm=TRUE) - mean(gp2.exo.MS$deltaPAS.MS,na.rm=TRUE)

#comparaison des delta entre M0 et M6  pour chaque groupe
t.test(gp1.exo.MS$deltaPAS.MS, gp2.exo.MS$deltaPAS.MS)  







############################################
#3.6:  Last Value Carried Forward (LOCF)####################################################################################
############################################

donnees.exo$pas.M3.locf <- ifelse(is.na(donnees.exo$pas.M3),donnees.exo$pas.M0,donnees.exo$pas.M3)
donnees.exo$pas.M6.locf <- ifelse(is.na(donnees.exo$pas.M6),donnees.exo$pas.M3.locf,donnees.exo$pas.M6)
donnees.exo$deltaPAS.locf <- donnees.exo$pas.M6.locf - donnees.exo$pas.M0

gp1.exo.locf <- subset(donnees.exo, donnees.exo$groupe=="Trt1")  #table spécifique au groupe 1
gp2.exo.locf <- subset(donnees.exo, donnees.exo$groupe=="Trt2")  #table spécifique au groupe 2


# PAS M0 groupe 1
mean(gp1.exo.locf$pas.M0,na.rm=TRUE)
sum(!is.na(gp1.exo.locf$pas.M0))
confEstimate(gp1.exo.locf$pas.M0)

# PAS M0 groupe2
mean(gp2.exo.locf$pas.M0,na.rm=TRUE)
sum(!is.na(gp2.exo.locf$pas.M0))
confEstimate(gp2.exo.locf$pas.M0)

# PAS M6 groupe 1
mean(gp1.exo.locf$pas.M6.locf,na.rm=TRUE)
sum(!is.na(gp1.exo.locf$pas.M6.locf))
confEstimate(gp1.exo.locf$pas.M6.locf)

# PAS M6 groupe 2
sum(!is.na(gp2.exo.locf$pas.M6.locf))
mean(gp2.exo.locf$pas.M6.locf,na.rm=TRUE)
confEstimate(gp2.exo.locf$pas.M6.locf)

# PAS delta groupe1
mean(gp1.exo.locf$deltaPAS.locf,na.rm=TRUE)
sum(!is.na(gp1.exo.locf$deltaPAS.locf))
confEstimate(gp1.exo.locf$deltaPAS.locf)

# PAS delta groupe2
mean(gp2.exo.locf$deltaPAS.locf,na.rm=TRUE)
sum(!is.na(gp2.exo.locf$deltaPAS.locf))
confEstimate(gp2.exo.locf$deltaPAS.locf)

# PAS delta de toutes les données
mean(gp1.exo.locf$deltaPAS.locf,na.rm=TRUE) - mean(gp2.exo.locf$deltaPAS.locf,na.rm=TRUE)

#comparaison des delta entre M0 et M6  pour chaque groupe
t.test(gp1.exo.locf$deltaPAS.locf, gp2.exo.locf$deltaPAS.locf) 









################################################################
#3.7:   Imputation par modèle de régression linéaire simple ####################################################################################
################################################################


#modèle de régression logistique simple
reglog <- lm(donnees.exo$pas.M3~donnees.exo$pas.M0, data= donnees.exo)
#summary pour récupérer l'intercept et Bêta 1
summary(reglog)

#imputation avec la formule obtenue grâce aux coefficients
donnees.exo$pas.M3.imputreglin  <- ifelse(is.na(donnees.exo$pas.M3),donnees.exo$pas.M0*1.00184 - 4.46943  ,donnees.exo$pas.M3)

#différence entre M3 imputé par régréssion linéaire simple et M0
donnees.exo$diffpasM3.reglin <- donnees.exo$pas.M3.imputreglin - donnees.exo$pas.M0 

#nouveau modèle de régression logistique simple
reglog2 <- lm(donnees.exo$pas.M6~donnees.exo$diffpasM3.reglin, data= donnees.exo)
summary(reglog2)

#imputation de M6 avec ce nouveau modèle
donnees.exo$pas.M6.imputreglin <- ifelse(is.na(donnees.exo$pas.M6),donnees.exo$diffpasM3.reglin*0.1511 + 153.4987  ,donnees.exo$pas.M6)

#création de la variable delta M1 M6 avec les NA imputées par régression linéaire
donnees.exo$deltaPAS.imputreglin <- donnees.exo$pas.M6.imputreglin  - donnees.exo$pas.M0

#création des tables pour chaque groupe
gp1.exo.imputreglin <- subset(donnees.exo, donnees.exo$groupe=="Trt1")  #table spécifique au groupe 1
gp2.exo.imputreglin <- subset(donnees.exo, donnees.exo$groupe=="Trt2")  #table spécifique au groupe 2

# PAS M0 groupe 1
mean(gp1.exo.imputreglin$pas.M0,na.rm=TRUE)
sum(!is.na(gp1.exo.imputreglin$pas.M0))
confEstimate(gp1.exo.imputreglin$pas.M0)

# PAS M0 groupe2
mean(gp2.exo.imputreglin$pas.M0,na.rm=TRUE)
sum(!is.na(gp2.exo.imputreglin$pas.M0))
confEstimate(gp2.exo.imputreglin$pas.M0)

# PAS M6 groupe 1
mean(gp1.exo.imputreglin$pas.M6.imputreglin ,na.rm=TRUE)
sum(!is.na(gp1.exo.imputreglin$pas.M6.imputreglin ))
confEstimate(gp1.exo.imputreglin$pas.M6.imputreglin )

# PAS M6 groupe 2
sum(!is.na(gp2.exo.imputreglin$pas.M6.imputreglin ))
mean(gp2.exo.imputreglin$pas.M6.imputreglin ,na.rm=TRUE)
confEstimate(gp2.exo.imputreglin$pas.M6.imputreglin )

# PAS delta groupe1
mean(gp1.exo.imputreglin$deltaPAS.imputreglin,na.rm=TRUE)
sum(!is.na(gp1.exo.imputreglin$deltaPAS.imputreglin))
confEstimate(gp1.exo.imputreglin$deltaPAS.imputreglin)

# PAS delta groupe2
mean(gp2.exo.imputreglin$deltaPAS.imputreglin,na.rm=TRUE)
sum(!is.na(gp2.exo.imputreglin$deltaPAS.imputreglin))
confEstimate(gp2.exo.imputreglin$deltaPAS.imputreglin)

#diff deux deltas
mean(gp1.exo.imputreglin$deltaPAS.imputreglin,na.rm=TRUE) - mean(gp2.exo.imputreglin$deltaPAS.imputreglin,na.rm=TRUE)

# PAS delta de toutes les données
mean(donnees.exo$deltaPAS.imputreglin,na.rm=TRUE)  
confEstimate(donnees.exo$deltaPAS.imputreglin)

#comparaison des delta entre M0 et M6  pour chaque groupe
t.test(gp1.exo.imputreglin$deltaPAS.imputreglin, gp2.exo.imputreglin$deltaPAS.imputreglin) 


################################################################
#3.8:   Imputation par modèle de régression linéaire multiple ####################################################################################
################################################################


#création de la variable BMI qui n'existait pas encore:
donnees.exo$bmi <- donnees.exo$poids/(donnees.exo$taille/100)^2 

#modèle de régression logistique multiple

#recodage de la variable groupe en variable binaire numérique
donnees.exo$groupe_recode  <- ifelse(donnees.exo$groupe == "Trt1",1  ,0)
donnees.exo$groupe_recode <- as.numeric(donnees.exo$groupe_recode)

reglog3 <- lm(donnees.exo$pas.M3~donnees.exo$pas.M0+donnees.exo$bmi+donnees.exo$age+donnees.exo$groupe_recode+donnees.exo$fumeur+donnees.exo$sexe, data= donnees.exo)
summary(reglog3)

#imputation avec la formule obtenue grâce aux coefficients
donnees.exo$pas.M3.multiple  <- ifelse(is.na(donnees.exo$pas.M3),donnees.exo$pas.M0*1.00308 + donnees.exo$bmi*-0.03589 + donnees.exo$age*0.03254 + donnees.exo$groupe_recode*-0.68088+ donnees.exo$fumeur*-0.62350 +donnees.exo$sexe*0.12569 -4.98648  ,donnees.exo$pas.M3)

#différence entre M3 imputé par régréssion linéaire multiple et M0
donnees.exo$diffpasM3.multiple <- donnees.exo$pas.M3.multiple - donnees.exo$pas.M0 

#nouveau modèle de régression logistique multiple 
reglog4 <- lm(donnees.exo$pas.M6~donnees.exo$diffpasM3.multiple+donnees.exo$bmi+donnees.exo$age+donnees.exo$groupe_recode+donnees.exo$fumeur+donnees.exo$sexe, data= donnees.exo)
summary(reglog4)

#imputation avec la formule obtenue grâce aux coefficients
donnees.exo$pas.M6.multiple  <- ifelse(is.na(donnees.exo$pas.M6),donnees.exo$diffpasM3.multiple*0.10357 + donnees.exo$bmi*0.06895 + donnees.exo$age*0.03470 + donnees.exo$groupe_recode* -3.11419+ donnees.exo$fumeur*-1.84687 +donnees.exo$sexe* -1.68891 +152.56833  ,donnees.exo$pas.M6)


#création de la variable delta M1 M6 avec les NA imputées par régression linéaire multiple
donnees.exo$deltaPAS.multiple <- donnees.exo$pas.M6.multiple - donnees.exo$pas.M0

#création des tables pour chaque groupe
gp1.exo.multiple <- subset(donnees.exo, donnees.exo$groupe=="Trt1")  #table spécifique au groupe 1
gp2.exo.multiple <- subset(donnees.exo, donnees.exo$groupe=="Trt2")  #table spécifique au groupe 2

# PAS M0 groupe 1
mean(gp1.exo.multiple$pas.M0,na.rm=TRUE)
sum(!is.na(gp1.exo.multiple$pas.M0))
confEstimate(gp1.exo.multiple$pas.M0)

# PAS M0 groupe2
mean(gp2.exo.multiple$pas.M0,na.rm=TRUE)
sum(!is.na(gp2.exo.multiple$pas.M0))
confEstimate(gp2.exo.multiple$pas.M0)

# PAS M6 groupe 1
mean(gp1.exo.multiple$pas.M6.multiple ,na.rm=TRUE)
sum(!is.na(gp1.exo.multiple$pas.M6.multiple ))
confEstimate(gp1.exo.multiple$pas.M6.multiple )

# PAS M6 groupe 2
sum(!is.na(gp2.exo.multiple$pas.M6.multiple ))
mean(gp2.exo.multiple$pas.M6.multiple ,na.rm=TRUE)
confEstimate(gp2.exo.multiple$pas.M6.multiple )

# PAS delta groupe1
mean(gp1.exo.multiple$deltaPAS.multiple,na.rm=TRUE)
sum(!is.na(gp1.exo.multiple$deltaPAS.multiple))
confEstimate(gp1.exo.multiple$deltaPAS.multiple)

# PAS delta groupe2
mean(gp2.exo.multiple$deltaPAS.multiple,na.rm=TRUE)
sum(!is.na(gp2.exo.multiple$deltaPAS.multiple))
confEstimate(gp2.exo.multiple$deltaPAS.multiple)

# PAS delta de toutes les données
mean(donnees.exo$deltaPAS.multiple,na.rm=TRUE)  
confEstimate(donnees.exo$deltaPAS.multiple)

#comparaison des delta entre M0 et M6  pour chaque groupe
t.test(gp1.exo.multiple$deltaPAS.multiple, gp2.exo.multiple$deltaPAS.multiple) 




