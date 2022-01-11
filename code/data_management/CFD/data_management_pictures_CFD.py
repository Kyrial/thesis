#!/usr/bin/env python
from os import listdir
from os.path import isfile, join, isdir
import shutil

path = "/home/nicolas/Bureau/internship_cefe_2021/data/original/CFD/Images"

sub_bases_folders = [f for f in listdir(path)]

for sub_base in sub_bases_folders:
    path2 = path + "/" + sub_base
    person_folder = [f for f in listdir(path2)]
    for person in person_folder:
        if isdir(path2 + "/" + person):
            path3 = path2 + "/" + person
            pictures_folder = [f for f in listdir(path3)]            
            for picture in pictures_folder:
                if picture == ".DS_Store":
                    continue
                attitude = picture.split("-")                
                if attitude[4] == "N.jpg":
                    shutil.copy(
                    path3 + "/" + picture, '/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/CFD/images')
        else:
            attitude = person.split("-")
            if attitude[4] == "N.jpg":  # si le visage est neutre
                shutil.copy(
                    path2 + "/" + person, '/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/CFD/images')

from os import rename

path = "/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/CFD/images"

files = [f for f in listdir(path)]

for file in files:
    parts = file.split("-")
    file2 = parts[0]+ "-"+ parts[1]+ "-"+ parts[2]+ "-"+ parts[4]  
    file1 = path + "/"+ file
    file2 = path + "/"+ file2
    rename(file1,file2)