#!/usr/bin/env python
from os import listdir
from os.path import isfile, join, isdir
import shutil

path_original = "/home/nicolas/Bureau/internship_cefe_2021/data/original/SCUT-FBP/Images"

path_redesigned = "/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/SCUT-FBP/images"

shutil.copytree(path_original, path_redesigned)
