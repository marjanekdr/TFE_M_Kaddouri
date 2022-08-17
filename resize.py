# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 11:07:31 2022

@author: marja
"""
from PIL import Image
import os
#Code permettant le redimensionnement de l'ensemble des images de la base de données
im_path = r"F:\IMAGE_RENAME_3_DIM/"
files= os.listdir(im_path)
path = r"F:\Resize_6/"
for i in range(len(files)):
    name = im_path + files[i]
    image = Image.open(name)
    width, height = image.size
    w8 = int(width/6)
    h8 = int(height/6)
    new_image8 = image.resize((w8,h8))
    name8 = path+ files[i][:-4]+".jpg"
    new_image8.save(name8)


    
    

    
