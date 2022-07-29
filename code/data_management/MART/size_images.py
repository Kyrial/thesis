#!/usr/bin/env python
from os import listdir
from os.path import isfile, join, isdir
import shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000


path = "/home/renoult/Bureau/thesis/data/redesigned/JEN/images"

sub_bases_folders = [f for f in listdir(path)]

prec_max = 0
prec_min = 9999999

for sub_base in sub_bases_folders:
    path2 = path + "/" + sub_base
    M = Image.open(path2)
    
    if M.size[0] > prec_max:
        prec_max= M.size[0]
    
    if M.size[0] < prec_min:
        prec_min= M.size[0]

print(prec_max)
print(prec_min)
    