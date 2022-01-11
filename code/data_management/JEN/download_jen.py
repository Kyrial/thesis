#!/usr/bin/env python
import urllib.request

path_links = '/home/nicolas/Bureau/internship_cefe_2021/data/original/JEN/File_link.txt'
path_names = '/home/nicolas/Bureau/internship_cefe_2021/data/original/JEN/File_Name.txt'

with open(path_links, 'r') as links:
    i = 0
    list_links = []
    for line in links:
        line2 = line[:-1]
        list_links.append(line2)

with open(path_names, 'r') as names:
    i = 0
    list_names = []
    for line in names:        
        line2 = line[:-1]
        list_names.append(line2)

i = 0
j = 0 #compteur d'échecs de téléchargements
list_fails = []

path_stock = '/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/JEN/images/'
for link in list_links:    
    path = path_stock + list_names[i] + '.jpg'
    try:  
        urllib.request.urlretrieve(link, path)
    except urllib.error.HTTPError:
        j += 1
        list_fails.append(list_names[i])

    i = i + 1

print(j,' images n ont pas été téléchargées') #65  images n ont pas été téléchargées

with open('/home/nicolas/Bureau/internship_cefe_2021/code/data_management/JEN/log_fails.txt',"w") as file:
    
    file.write('il y a eu')
    file.write(j)
    file.write('échecs, les voici: ')
    file.write(list_fails)
