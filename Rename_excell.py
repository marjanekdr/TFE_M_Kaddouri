# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:07:31 2022

@author: marja
"""

#library 
import pandas as pd 
import shutil
import os


#ouverture fichier csv
df = pd.read_csv(r"D:\tfe\code\nom fichiers/DATA_VAL2.csv",sep =";")


#rename and copy             
        

path = r"F:\IMAGE_RENAME_3"
files= os.listdir(path)



for i in range(len(df)):
    if (df['NOUVEAU_NOM']+".jpg")[i] not in files:
        shutil.copy(df['name'][i],os.path.join(path,df['NOUVEAU_NOM'][i]+".jpg"))
        