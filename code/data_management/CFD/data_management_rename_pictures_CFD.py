#!/usr/bin/env python
from os import listdir
from os.path import isfile, join, isdir
from os import rename

path = "/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/CFD/images"

files = [f for f in listdir(path)]

for file in files:
    parts = file.split("-")
    file2 = parts[0]+ "-"+ parts[1]+ "-"+ parts[2]+ "-"+ parts[4]  
    file1 = path + "/"+ file
    file2 = path + "/"+ file2
    rename(file1,file2)

