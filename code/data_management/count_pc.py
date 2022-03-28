#!/usr/bin/env python
from os import listdir
import csv

dbs = ["MART", "JEN", "CFD"]


for db in dbs:

    print("#################")
    print(db)

    path = "/home/renoult/Bureau/thesis/results/" + db + "/pca"

    files = [f for f in listdir(path)]
   
    for file in files:

        print(file)
 
        reader = csv.reader(open(path+"/"+file))
        print(len(next(reader)))
        
       
