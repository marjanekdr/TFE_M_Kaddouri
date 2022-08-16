# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:46:41 2022

@author: marja
"""

import pandas as pd 
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np


#open csv

df = pd.read_csv("D:/TFE/CODE/NOM FICHIERS/nomfichier_nettoye.csv",sep =",")
df2 = pd.read_csv(r"D:\TFE\CODE\NOM FICHIERS\images_nom_finam/NOMFICHIER_EXCELL_2.csv",sep =";")
classes = pd.read_csv(r"D:\TFE\CODE\NOM FICHIERS\Classes\VALEUR_ESPECE.csv",sep =";")


#Creation d'un dictionnaire de classes 

dico = {v:k for k,v in enumerate(classes["Espece"])}

df["Classe"] = 0
for i in range(len(df)):
        df["Classe"][i] = dico.get(df["name"][i][22:26] )
 
for i in range(len(df)):
    df["name"][i] = df["name"][i][22:-4]
 
df["ID"] = 0
for i in range(len(df)): 
    for j in range(len(df2)):
        if df["name"][i] == df2["NOUVEAU_NOM"][j]:
            df["ID"][i] = df2["ID"][j]

# Classification des données en entrainement validation et test

#Classification des données en Entrainement, Validation et Test 
cv = StratifiedGroupKFold( n_splits = 5)
X=df["name"]
y=df["Classe"]
groups=df["ID"]


for train_index, test_index in cv.split(X=df["name"], y=df["Classe"], groups=df["ID"]):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    groups_train, groups_test = groups[train_index], groups[test_index]
df_test = pd.DataFrame(X_test)
df_test["classe"]=y_test
df_test["ind"]=groups_test
df_train=pd.DataFrame(X_train)
df_train["classe"]=y_train
df_train["ind"]=groups_train
df_test.to_csv("D:/TFE/CODE/NOM FICHIERS/DATA_TEST.csv", index = False)  
df_train.to_csv("D:/TFE/CODE/NOM FICHIERS/DATA_TRAIN.csv", index = False)     

df_train = pd.read_csv("D:/TFE/CODE/NOM FICHIERS/DATA_TRAIN.csv",sep =",")
cv2 = StratifiedGroupKFold( n_splits = 8)
X=df_train["name"]
y=df_train["classe"]
groups=df_train["ind"]
for train_index, test_index in cv2.split(X=df_train["name"], y=df_train["classe"], groups=df_train["ind"]):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    groups_train, groups_test = groups[train_index], groups[test_index]
df_test = pd.DataFrame(X_test)
df_test["classe"]=y_test
df_test["ind"]=groups_test
df_train=pd.DataFrame(X_train)
df_train["classe"]=y_train
df_train["ind"]=groups_train
df_test.to_csv("D:/TFE/CODE/NOM FICHIERS/DATA_VAL.csv", index = False)  
df_train.to_csv("D:/TFE/CODE/NOM FICHIERS/DATA_TRAIN.csv", index = False)                    


