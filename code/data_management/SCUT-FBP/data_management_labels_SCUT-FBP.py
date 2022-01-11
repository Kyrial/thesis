#!/usr/bin/env python

from os import listdir
import xlrd
import csv
import numpy as np

#traitement du fichier excel

excel_labels = xlrd.open_workbook("/home/nicolas/Bureau/internship_cefe_2021/data/original/SCUT-FBP/All_Ratings.xlsx")

sheet = excel_labels.sheet_by_index(0)

#dictionnaire avec les ID en clé, contenant une liste des différentes notes

data = {}

for r in range (1, sheet.nrows-1):
    if sheet.cell_value(rowx = r, colx = 1) in data: #si l'ID est déja dans le dictionnaire
        data[sheet.cell_value(rowx = r, colx = 1)].append(int(sheet.cell_value(rowx = r, colx = 2)))
    else: #si l'ID n'est pas encore dans le dictionnaire
        data[sheet.cell_value(rowx = r, colx = 1)] = []
        data[sheet.cell_value(rowx = r, colx = 1)].append(int(sheet.cell_value(rowx = r, colx = 2)))


data_means = {}

for key in data:
    data_means[key] = np.mean(data[key])

with open('/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv', 'w') as f:
    for key in data_means.keys():
        f.write("%s,%s\n"%(key,data_means[key]))