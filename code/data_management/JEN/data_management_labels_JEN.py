#!/usr/bin/env python

from os import listdir
import xlrd
xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True
import csv
import numpy as np

#traitement du fichier excel
excel_labels = xlrd.open_workbook("/home/nicolas/Bureau/internship_cefe_2021/data/original/JEN/Subjective scores.xlsx")
sheet = excel_labels.sheet_by_index(0)

#liste des images téléchargées
path = "/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/JEN/images"
files = [f for f in listdir(path)]



print(len(files))
#dictionnaire avec les ID en clé, note en valeur
data = {}
i = 0

for r in range (1, sheet.nrows):
    if sheet.cell_value(rowx = r, colx = 0) in files:
        i += 1
        data[sheet.cell_value(rowx = r, colx = 0)] = int(sheet.cell_value(rowx = r, colx = 2))  #beauty  

print(i)

with open('/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/JEN/labels_JEN.csv', 'w') as f:
    for key in data.keys():
        f.write("%s,%s\n"%(key,data[key]))