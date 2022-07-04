#!/usr/bin/env python
from os import listdir
from os.path import isfile, join, isdir
import shutil

import csv

path = "/home/renoult/Bureau/thesis/results/CFD/log__CFDwithlabels_imagenet_gini_flatten__BRUTMETRICS.csv"


f = open(path)

myReader = csv.reader(f)
for row in myReader:
    if row[0][4:6] == 'BF':
        print(row[0])

for row in myReader:
    if row[0][6] == 'F':
        print(row[0])