#!/usr/bin/env python

###############################################
###################IMPORTS#####################
###############################################

import os
import shutil
from PIL import Image

###############################################
###################PARAMETRES##################
###############################################

nb_class = 30 #nombre de classes
nb_pictures = 30 #nombre d'images par classe

###############################################
###################DESCRIPTION#################
###############################################

#1. pour chaque classe
        

        # - vérifier que les images ne sont pas corrompues, si corrompu, supprimer l'image (est-ce stocké dans un fichier de log? sinon comment le tester?)

        # - compter les image: 
                                # si < 100, supprimer la classe 
                                # si >= 100, supprimer n - 100 images 

#on a maintenant plus que des classes de taille 100

#2. en prendre 100, supprimer les autres

#Par classe, ne pas avoir un fichier d'images et deux autres trucs inutiles (un csv et un xml) mais directement toutes les images

###############################################
###################CODE########################
###############################################

#path
path = "/home/renoult/Bureau/part-of-imagenet/partial_imagenet"

#liste des classes
class_names = [f for f in os.listdir(path)]

##################################################
# 1. On veut 100 images non corrompues par classes
##################################################

i = 0

#pour chaque classe
for each in class_names:

        #path des images de cette classe
        path_class_pictures = "/home/renoult/Bureau/part-of-imagenet/partial_imagenet" + "/" + each +"/Images"

        #path du contenu de la classe (XML, images, csv)
        path_class = "/home/renoult/Bureau/part-of-imagenet/partial_imagenet" + "/" + each

        #liste du nom des images de la classe courante
        pictures = [f for f in os.listdir(path_class_pictures)]    
        

        #pour chaque image de la classe courante
        for picture in pictures:
                
                #path de l'image courante de la classe courante
                path_picture = path_class_pictures + "/" + picture

                #vérifier que les images ne sont pas corrompues

                #vérifier qu'on peut ouvrir l'image
                try:
                        im=Image.open(path_picture)                                             
             
                #sinon la supprimer
                except IOError:                                               
                        os.remove(path_picture)

        #on reliste les images pour prendre en compte celles qui ont été supprimées
        pictures = [f for f in os.listdir(path_class_pictures)] 
         

        #suppression des répertoires ou il y a moins de nb_pictures images
        #si il y a moins de nb_pictures images dans le répertoire        
        if len(pictures) < nb_pictures:                
                #on supprimer le répertoire de la classe courante et on passe a la classe suivante
                shutil.rmtree(path_class)    
                continue        

        #ici, on est donc dans le cas ou il y a nb_pictures ou plus images dans la classe
        #On n'en veut que nb_pictures, on supprime donc n images avec n = len(pictures)-100
        n = len(pictures) - nb_pictures

        #on supprime les n premières images de la classe
        for picture in pictures[:n]:
                os.remove(path_class_pictures + "/" + picture)       

####################################################################
# 2. On veut 100 classes parmi les n classes de 100 images restantes
####################################################################
#on recompte pour prendre en compte celles qui ont été supprimées
class_names = [f for f in os.listdir(path)]

#on calcule le nombre de classes a supprimer (les n qui dépassent en conservant 100 classes)
n = len(class_names) - nb_class

#On supprime ces n premières classes
all_class = os.listdir(path)
for each_class in all_class[:n]:        
        shutil.rmtree("/home/renoult/Bureau/part-of-imagenet/partial_imagenet" + "/" + each_class)


####################################################################
# 4. Un peu de nettoyage: dans chaque classe, ily a un subfolder /Images avec les images dedans, et un csv. 
#On en veut que les images, et directement dans le folder de la classe
####################################################################

#on récupère de nouveau le nom des classes restantes
all_class = os.listdir(path)

#pour chaque classe
for each_class in all_class:      
        
        

        #4.1: suppresion du .csv

        #liste des fichiers dans chaque classe
        files_in_directory = os.listdir("/home/renoult/Bureau/part-of-imagenet/partial_imagenet" + "/" + each_class)
        #séléction du .csv
        filtered_files = [file for file in files_in_directory if file.endswith(".csv")]        
        #suppression du csv (ou des csv, au cas ou il y en ai plusieurs)
        for file in filtered_files:
                path_to_file = os.path.join("/home/renoult/Bureau/part-of-imagenet/partial_imagenet" + "/" + each_class, file)
                os.remove(path_to_file)

        #4.2: déplacement des images su subfolder /images au folder de la classe

        # Define the source and destination path
        source = "/home/renoult/Bureau/part-of-imagenet/partial_imagenet" + "/" + each_class + "/Images"
        destination = "/home/renoult/Bureau/part-of-imagenet/partial_imagenet" + "/" + each_class

        # code to move the files from sub-folder to main folder.
        files = os.listdir(source)
        for file in files:
                file_name = os.path.join(source, file)
                shutil.move(file_name, destination)
        
        #Suppression du folder /Images désormais vide, et du folder d'annotations en XML (/Annotations)
        shutil.rmtree("/home/renoult/Bureau/part-of-imagenet/partial_imagenet" + "/" + each_class + "/" + "Images")
        shutil.rmtree("/home/renoult/Bureau/part-of-imagenet/partial_imagenet" + "/" + each_class + "/" + "Annotation")

