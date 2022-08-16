# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 17:07:20 2022

@author: marja
"""

"""
Created on Fri Jun  3 13:19:28 2022

@author: marja
"""

"""
Created on Thu Jun  2 11:25:15 2022

@author: gef
"""

# ETAPE 1 - importation des librairies nécessaires
import os
# PyTorch
import torch
import torchvision as tv
# NumPy
import numpy as np
import cv2 as cv 

# Matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import pandas as pd 
from PIL import Image
# Time
import time

import wandb
from sklearn.metrics import confusion_matrix, classification_report
import albumentations as A
from albumentations import Normalize 
from albumentations.pytorch.transforms import ToTensorV2
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau


#_________________________________________________________________________________
# ETAPE 2 - Sélection du processeur de calcul = cuda
#-------------------------------------------------------------
use_cuda = torch.cuda.is_available()
if not use_cuda:
  print("WARNING: PYTORCH COULD NOT LOCATE ANY AVAILABLE CUDA DEVICE.\n\n" \
        "  ...make sure you have enabled your notebook to use a GPU!" \
        "  (Edit->Notebook Settings->Hardware Accelerator: GPU)")
else:
  print("All good, a GPU is available.")
#_________________________________________________________________________  
#ETAPE 3 - Création d'un seed permettant de pouvoir répéter des actions de manière similaire
def set_seed(seed):
  # CPU variables
  random.seed(seed) 
  np.random.seed(seed)
  # Python
  torch.manual_seed(seed) 
  # GPU variables
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True 
  # torch.set_deterministic(True)
  torch.backends.cudnn.benchmark = False
  
#__________________________________________________________________________________________________
#ETAPE 4 - nomination des fichiers de sorties


name_model= "07_07_model_augmentation__schedmax_batch16"
path_conf = r"D:\Marjane\Resultats\07_07_model_augm__schedmax_batch_16/"
#__________________________________________________________________________________________________
# ETAPE 5 - Création de la classe d'ouvertude des données d'entrée

class ImageCSVFolder(Dataset):

  def __init__(self, img_dir, csv_file, transform=None):
      self.csv_file = pd.read_csv(csv_file,delimiter=';',index_col=False,header=0)
      self.img_dir = img_dir
      self.transform = transform
      self.labels = self.csv_file["classe"].tolist()
      self.names= self.csv_file["name"].tolist()

  def __len__(self):
      return len(self.labels)

  def __getitem__(self, idx):
      # img_path = os.path.join(img_dir, self.csv_file[idx, 0] + ".jpg")
      img_path = os.path.join(self.img_dir, self.names[idx] + ".jpg")
      image = Image.open(img_path)
      image = np.array(image)

      label = self.labels[idx]
      name = self.names[idx]

      if self.transform:
          transform_pipeline = A.Compose(self.transform)
          image = transform_pipeline(image=image)["image"]

      return image, label, name
#__________________________________________________________________________________________________
# ETAPE 6 - Définition des classes

classes = pd.read_csv(r"D:\Marjane/VALEUR_ESPECE.csv",sep =";")
class_names = list(classes["Espece"])
print(class_names)
#STEP 2 : Ouverture des fichiers 

csv_file_train = r"D:\Marjane/DATA_TRAIN2.csv"
csv_file_val = r"D:\Marjane/DATA_VAL2.csv"

img_dir = r"D:\Marjane\Resize_8/"

#__________________________________________________________________________________________________
#ETAPE 7 - Fonction de transformation 

transform =[ A.augmentations.transforms.Transpose(p=1),
    A.augmentations.transforms.Flip(p=0.2),  
    A.augmentations.geometric.rotate.Rotate(limit=90, border_mode=cv.BORDER_CONSTANT, value=(0,0,0), p= 0.2),
    #A.augmentations.geometric.transforms.ElasticTransform( alpha=0, sigma= 77, alpha_affine=32, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], approximate=False,p=0.2),
    A.augmentations.transforms.PadIfNeeded (min_height=578, min_width=578,border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
    Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2()]


trans =[ A.augmentations.transforms.Transpose(p=1),
    A.augmentations.transforms.PadIfNeeded (min_height=578, min_width=578, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
    Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2()]

#__________________________________________________________________________________________________
#ETAPE 8 : définition du seed et ouverture des fichiers
set_seed(1)


df_train = ImageCSVFolder(img_dir= img_dir, csv_file= csv_file_train, transform = transform)
df_val = ImageCSVFolder(img_dir= img_dir, csv_file= csv_file_val, transform = trans)

#__________________________________________________________________________________________________
# ETAPE 9 :  Regroupement des images en mini-batch (taille = 8)

batch_size = 8
batch = 1
train_loader = torch.utils.data.DataLoader(dataset=df_train, batch_size=batch_size, num_workers=0, shuffle = True) 
valid_loader = torch.utils.data.DataLoader(dataset=df_val, batch_size=batch, num_workers=0, shuffle = False)

#__________________________________________________________________________________________________
# ETAPE 10 : Visualisation des mini-batches
sample_images, sample_labels, _ =  next(iter(train_loader))
print("Shapes :")
print("-------------------------------------------------")
print(f"Images tensor : {sample_images.shape}")  # BxCxHxW
print(f"Labels tensor : {sample_labels.shape}")  # Bx1  (une étiquette par image du minibatch)
print("-------------------------------------------------")

# Nombres d'images affichées
display_batch_size = min(8, batch_size)
fig = plt.figure(figsize=(16, 3))
for ax_idx in range(display_batch_size):
  print(int(sample_labels[ax_idx]))
  # Ajout d'une image à côté de la précédente
  ax = fig.add_subplot(1, 8, ax_idx + 1)
  # Désactiver les grilles
  ax.grid(False)
  # Graduation des axes x et y (aucune, list vide)
  ax.set_xticks([])
  ax.set_yticks([])
  # Ajout du nom de la classe en titre
  class_name=class_names[sample_labels[ax_idx]]
  ax.set_title(class_name)
  print(class_name)
  # Tenseur correspondant à l'image
  display = sample_images[ax_idx, ...].numpy()
  # Transposition : (C,H,W) => (H,W,C) (tel que demandé par matplotlib)
  display = display.transpose((1, 2, 0)) 
  # Inversion de la normalisation 
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])  
  display = std * display + mean  # on inverse la normalisation
  # Elimination des valeurs qui sortent de l'intervalle d'affichage
  display = np.clip(display, 0, 1) 
  # Affichage de l'image
  plt.imshow(display)
plt.show()

# ETAPE 10 - Définition du modèle
#-------------------------------------------------------------
# Utilisations d'un modèle préentrainé (ImageNet datasets) : 
# ResNet-18 (torchvision)
valid_acc_max = 0.0
model = tv.models.resnet18(pretrained=True)
num_backbone_features = model.fc.in_features
model.fc =torch.nn.Sequential(torch.nn.Linear(in_features=num_backbone_features, out_features=num_backbone_features, bias= True),
                              torch.nn.Linear(in_features=num_backbone_features, out_features=len(class_names), bias=True))
print(model)


# ETAPE 11 - Définition de la fonction de coût et de l'optimiseur
#-------------------------------------------------------------
# Taux d'apprentissage (contrôle de la taille du pas de mise à jour des paramètres)
learning_rate = 1e-4

# Momentum (ou inertie) de l'optimiseur (ici SGD)
momentum = 0.9  

# Pénalité de régularisation (L2) sur les paramètres
#On force le modèle à prendre seulement les valeurs les plus petites en ajoutant une fonction de coût à avoir des valeurs larges à la loss function
weight_decay = 1e-7  

# Nombre d'epochs consécutives avec taux d'apprentissage fixe 
#lr_step_size = 50 

# Facteur de réduction du taux d'apprentissage, après lr_step_size
lr_step_gamma = 0.1  

# Instanciation de la fonction de coût sous forme d'un objet
#Creation d'un dictionnaire de classes 

dico = {v:k for k,v in enumerate(classes["Espece"])}
 # specify loss function (categorical cross-entropy)
all_labels = list(df_train.csv_file["classe"])
effectif = []
print(dico)
for (Key,Value) in dico.items():
    effectif.append(all_labels.count(Value))

effectif_max = max(effectif)
weigths = [effectif_max/eff for eff in effectif]

weigths=torch.cuda.FloatTensor(weigths)

criterion = torch.nn.CrossEntropyLoss(weight= weigths)

# Instanciation de l'optimiseur (SGD); on lui fournit les paramètres du modèle qui nécessitent une MàJ
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# Instanciation du 'Scheduler' permettant de modifier le taux d'apprentissage en fonction de l'epoch
scheduler = ReduceLROnPlateau(optimizer, 'max',patience=4,min_lr = 0.000001,verbose = True)

if use_cuda:
  model = model.cuda()
  
#ETAPE 12 : initiation wandb

#WANDB
wandb.init(project="TFE", entity="mrjnk")

wandb.config.lr = learning_rate
wandb.config.nbr_classes = len(class_names)
wandb.config.optimizer = optimizer
wandb.config.scheduler = scheduler
wandb.config.loss_fucntion = "CrossEntropyLoss"
wandb.config.batch_size = batch_size
wandb.config.model = "Resnet18"
wandb.config.transformation = str(transform)

wandb.config.model_name = "Resnet18_1_2_06"
wandb.config.size_dataset = len(train_loader.sampler)+len(valid_loader.sampler)

  
# Etape 13 : Entrainement du modèle
#-------------------------------------------------------------
# Pour être certain qu'on démarre avec un modèle "vide"
# model.load_state_dict(model_init_state)

# Nombre d'epochs totale pour l'entraînement (à ajuster au besoin)
epochs = 50  

# Variables pour affichage des évolutions
train_losses, valid_losses = [], [] 
train_accuracies, valid_accuracies = [], [] 
# Variables pour le test final du meilleur modèle
best_model_state, best_model_accuracy = None, None  

last_print_time = time.time()

for epoch in range(epochs):
    
  #-------------------------------------------------------------------------
  # Première étape: on utilise le 'train_loader' pour entraîner le modèle
  #-------------------------------------------------------------------------
  # On va accumuler les coûts pour en afficher la courbe
  train_loss = 0  
  valid_loss = 0
  # On va aussi accumuler les bonnes/mauvaises classifications pour calculer
  # l'accuracy du modèle 
  train_correct, train_total = 0, 0 
  valid_correct,valid_correct=0,0
  # Mise du modèle en mode "entraînement" (utile pour certaines couches...)
  model.train()  
  
  # Chaque itération sur un 'DataLoader' produira un minibatch de données prétraitées
  for batch_idx, minibatch in enumerate(train_loader):
    
    if time.time() - last_print_time > 10:
      last_print_time = time.time()
      print(f"\ttrain epoch {epoch+1}/{epochs} @ iteration {batch_idx+1}/{len(train_loader)}...")
    
    # Dans ce cas-ci, le 'Dataset' défini plus tôt charge les images en tuples (image, label)
    images = minibatch[0]  # rappel: en format (B,C,H,W)
    labels = minibatch[1]  # rappel: en format (B,1)
    
    if use_cuda:
      # si nécessaire, on transfert nos données vers le GPU (le modèle y est déjà)
      images = images.cuda()
      labels = labels.cuda()
      
    # Il faut remettre les gradients à zéro après chaque itération (avec un minibatch)
    optimizer.zero_grad()
    
    # On calcule la prédiction du modèle en lui donnant le tenseur des images uniquement
    preds = model(images)
    
    # On calcule la perte à partir des prédictions et des étiquettes réelles
    # Rappel : criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(preds, labels)

    
    # On utilise la valeur de perte pour rétropropager le gradient à travers le réseau
    loss.backward()
    
    # Une fois les gradients à jour, on demande à l'optimiseur de modifier les poids
    optimizer.step()
    
    # Rafraîchissement des métriques d'entraînement
    train_loss += loss.item()*(minibatch[0].size(0))
    # la fonction '.item()' retourne un scalaire à partir du tenseur
    train_correct += (preds.topk(k=1, dim=1)[1].view(-1) == labels).nonzero().numel()
    train_total += labels.numel()
  
  # On calcule les métriques globales pour l'epoch (loss, accuracy, recall)
  train_loss = train_loss / len(train_loader.dataset)
  train_losses.append(train_loss)
  train_accuracy = train_correct / train_total
  train_accuracies.append(train_accuracy)
  


  # Optional
  wandb.watch(model)
  last_print_time = time.time()
  print(f"train epoch {epoch+1}/{epochs}: loss={train_loss:0.4f}, accuracy={train_accuracy:0.4f}")
  
  #-------------------------------------------------------------------------
  # Deuxième étape: on utilise le 'valid_loader' pour évaluer le modèle
  #-------------------------------------------------------------------------
  # On va accumuler les coûts pour en afficher la courbe

  # On va aussi accumuler les bonnes/mauvaises classifications pour calculer
  # l'accuracy du mo dèle
  
  valid_loss = 0.0 
  valid_correct, valid_total = 0, 0

  y_pred = []
  y_true = []

  # Mise du modèle en mode "évaluation" (utile pour certaines couches...)
  model.eval()  
  with torch.no_grad():
      
      # Boucle semblable à celle d'entraînement, mais on utilise l'ensemble de validation
      for batch_idx, minibatch in enumerate(valid_loader):
        
        if time.time() - last_print_time > 10:
          last_print_time = time.time()
          print(f"\tvalid epoch {epoch+1}/{epochs} @ iteration {batch_idx+1}/{len(valid_loader)}...")
        
        images = minibatch[0] 
        labels = minibatch[1]
        
    
        if use_cuda:
          images2 = images.cuda()
          labels2 = labels.cuda()
        
        # Ici, on n'a plus besoin de l'optimiseur, on cherche seulement à évaluer
         # utile pour montrer explicitement qu'on n'a pas besoin des gradients
        preds2 = model(images2)
        loss2 = criterion(preds2,labels2)
        valid_loss += loss2.item()*(minibatch[0].size(0))
        valid_correct += (preds2.topk(k=1, dim=1)[1].view(-1) == labels2).nonzero().numel()
        valid_total += labels2.numel()
      # On calcule les métriques globales pour l'epoch
      
        valid_accuracy = valid_correct / valid_total
        valid_accuracies.append(valid_accuracy)
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_losses.append(valid_loss)
       
        #wandb.watch(model)
        _,pred=torch.max(preds2,1)
        y_pred.append(int(pred))
        y_true.append(labels2.item())
  
      # On mémorise les poids si le modèle surpasse le meilleur à date
      if best_model_accuracy is None or valid_accuracy > best_model_accuracy:
        best_model_state = model.state_dict()
        best_model_accuracy = valid_accuracy
      
      last_print_time = time.time()
      print(f"valid epoch {epoch+1}/{epochs}: loss={valid_loss:0.4f}, accuracy={valid_accuracy:0.4f}")  
      print("----------------------------------------------------\n")
   # save model if validation loss has decreased
  
  
  wandb.log({"train_loss": train_loss, "valid_loss" : valid_loss, "valid accuracy" : valid_accuracy})
  scheduler.step(valid_accuracy)
       
  if valid_accuracy >= valid_acc_max:
    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
    valid_acc_max,
    valid_accuracy))
    torch.save(model.state_dict(), os.path.join(r'D:\Marjane/model',name_model))
    valid_acc_max = valid_accuracy
    best_epoch = epoch
    
# ETAPE 14 - Evaluation du meilleur modèle
#-------------------------------------------------------------
# On accumule les bonnes/mauvaises classifications pour le test



mat_conf = confusion_matrix(y_true = y_true,y_pred = y_pred)
conf = np.zeros((len(class_names)+1,len(class_names)+2),dtype=object)
conf[len(conf)-1,0] = 'total'
for i in range(len(class_names)):
    conf[i,0] = class_names[i]
    for j in range(len(class_names)):
            conf[i,j+1] = mat_conf[i,j]
            # print(mat_conf[i,j])
            conf[i,len(class_names)+1] += mat_conf[i,j]
            conf[len(class_names),len(class_names)+1] += mat_conf[i,j]
            conf[len(class_names),j+1] += mat_conf[i,j]
columns = []
columns.append('label/predict')
columns = columns + class_names
columns.append('total')


df_cm = pd.DataFrame(conf,columns=columns)
print(df_cm)

path_conf = path_conf + "confusionMatrix.csv"
df_cm.to_csv(path_conf, index = False)     
# # STEP 11 - Affichage des courbes de performances
# #-------------------------------------------------------------
# x = range(1, epochs + 1)

# fig = plt.figure(figsize=(12, 4))

# ax = fig.add_subplot(1, 2, 1)
# ax.plot(x, train_losses, label='train')
# ax.plot(x, valid_losses, label='valid')
# ax.set_xlabel('# epochs')
# ax.set_ylabel('Loss')
# ax.legend()

# ax = fig.add_subplot(1, 2, 2)
# ax.plot(x, train_accuracies, label='train')
# ax.plot(x, valid_accuracies, label='validation')

# ax.set_xlabel('# epochs')
# ax.set_ylabel('Accuracy')
# ax.legend()


# plt.show()


#____________________________________________________________________________________________
#ETAPE 14 : TEST
set_seed(1)
#ouveture du dossier
batch = 1
trans =[ A.augmentations.transforms.Transpose(p=1),
    A.augmentations.transforms.PadIfNeeded (min_height=578, min_width=578, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
    Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2()]
csv_file_test ="D:\Marjane/DATA_TEST2.csv"
df_test = ImageCSVFolder(img_dir= img_dir, csv_file= csv_file_test, transform = trans)
test_loader = torch.utils.data.DataLoader(dataset=df_test, batch_size=batch, num_workers=0, shuffle = False)

csv_file_heatmap = r"D:\Marjane/erreur/list_erreur.csv"
df_heat = ImageCSVFolder(img_dir= img_dir, csv_file= csv_file_heatmap, transform = trans)
heat_loader = torch.utils.data.DataLoader(dataset=df_heat, batch_size=batch, num_workers=0, shuffle = False)
#lancement du dossier 

opt_model ="D:\Marjane\model/18_06_model_augmentation"
model.load_state_dict(torch.load(opt_model))
model = model.cuda()
from captum.attr import Occlusion
from captum.attr import visualization as viz
from torchvision.transforms import functional


test_loss = 0.0
test_correct, test_total = 0,0
test_accuracies = []
test_losses = []
y_pred = []
y_true = []
y_score = []
h = 0
occlusion = Occlusion(model)
model.eval()  
with torch.no_grad():
    
    # Boucle semblable à celle d'entraînement, mais on utilise l'ensemble de validation
    for batch_idx, minibatch in enumerate(heat_loader):
      
      
      if time.time() - last_print_time > 10:
        last_print_time = time.time()
        print(f"\ @ iteration {batch_idx+1}/{len(test_loader)}...")
      
      images = minibatch[0] 
      labels = minibatch[1]
      name = minibatch[2]
      target2 = labels.cpu().numpy()[0]
  
      if use_cuda:
        images2 = images.cuda()
        labels2 = labels.cuda()
      
      # Ici, on n'a plus besoin de l'optimiseur, on cherche seulement à évaluer
       # utile pour montrer explicitement qu'on n'a pas besoin des gradients
      preds2 = model(images2)
      loss2 = criterion(preds2,labels2)
      test_loss += loss2.item()*(minibatch[0].size(0))
      test_correct += (preds2.topk(k=1, dim=1)[1].view(-1) == labels2).nonzero().numel()
      test_total += labels2.numel()
    # On calcule les métriques globales pour l'epoch
    
      test_accuracy = test_correct / test_total
      test_accuracies.append(test_accuracy)
      test_loss = test_loss / len(test_loader.dataset)
      test_losses.append(test_loss)
     
      #wandb.watch(model)
      
      confid = torch.softmax(preds2, dim=1).max()
      
      _,pred=torch.max(preds2,1)
      pred3 = pred.cpu().numpy()[0]
      y_pred.append(int(pred))
      y_true.append(labels2.item())
      y_score.append(confid.item())

    # On mémorise les poids si le modèle surpasse le meilleur à date
      if h%25 == 0:#target2 != pred2
        data10 = images2
        target10=int(pred3) 
        print("Label =",target2)
        print("Prédiction =",pred3)
        
        strides = (3, 25, 25)                                
        sliding_window_shapes=(3,50, 50)
        baselines = 0
        
        attribution = occlusion.attribute(data10,
                                            strides = strides,
                                            target=target10,
                                            sliding_window_shapes=sliding_window_shapes,
                                            baselines=baselines)
        
        data11 = functional.normalize(data10, [-0.485/0.229, -0.456/0.224, -0.406/0.225],[1/0.229, 1/0.224, 1/0.225]) 
        x = viz.visualize_image_attr_multiple(np.transpose(attribution.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(data11.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              ["original_image", "heat_map"],
                                              ["all", "positive"],
                                              ["image", "attribution"],cmap = "Reds",
                                              show_colorbar=True,
                                              outlier_perc=2,
                                             )
    last_print_time = time.time()
    # CAPTUM
   
#___________________________________________________________________________________________________________________________________
#ETAPE 15 Création et mise en page de la matrice de confusion

mat_conf = confusion_matrix(y_true = y_true,y_pred = y_pred)
conf = np.zeros((len(class_names)+1,len(class_names)+2),dtype=object)
conf[len(conf)-1,0] = 'total'
for i in range(len(class_names)):
    conf[i,0] = class_names[i]
    for j in range(len(class_names)):
            conf[i,j+1] = mat_conf[i,j]
            # print(mat_conf[i,j])
            conf[i,len(class_names)+1] += mat_conf[i,j]
            conf[len(class_names),len(class_names)+1] += mat_conf[i,j]
            conf[len(class_names),j+1] += mat_conf[i,j]
columns = []
columns.append('label/predict')
columns = columns + class_names
columns.append('total')


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
     cmap = plt.get_cmap('OrRd')

    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation='nearest', cmap= cmap)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45,ha="right")
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm.round(2)
      


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
class_legende = ["Afzelia bipindensis", "Afrostyrax lepidophyllus", "Alstonia boonei", "Annickia affinis", "Corynanthe pachyceras", "Desbordesia glaucescens",
              "Entandrophragma cylindricum","Irvingia gabonensis", "Keayodendron bridelioides", "Meiocarpidium lepidotum","Pericopsis elata",
              "Polyalthia suaveolens", "Pterocarpus soyauxii", "Rauvolfia macrophylla","Strombosia pustulata","Strombosiopsis tetrandra", "Tabernaemontana crassa",
              "Terminalia superba","Trichilia welwitschii", "Xylopia aethiopica"]
plot_confusion_matrix(mat_conf,class_legende, title='Matrice de confusion', 
                          cmap=None, normalize=False)
    

    
df_cm = pd.DataFrame(conf,columns=columns)
print(df_cm)
path_conf = "D:/Marjane/Res_final/confusionMatrix.csv"
df_cm.to_csv(path_conf, index = False)  

#ETAPE 16 : Classifation report 
report = classification_report(y_true, y_pred, output_dict=True)
report = pd.DataFrame(report).transpose()
Path_rep = "D:/Marjane/Res_final/summary.csv"
report.to_csv(Path_rep, index = True)

import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_classification_report(classificationReport,
                               title='Classification report',
                               cmap='RdBu'):

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[1:]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.tight_layout()



plot_classification_report(report)


df = pd.read_csv(r"D:\Marjane/DATA_TEST2.csv",sep =";")
df_result = pd.DataFrame(columns= ['Nom', "label", "prédiction", "pourcentage", "TP", "FP"])
df_result["Nom"] = df["name"] 
df_result["prédiction"] = y_pred
df_result["pourcentage"] = y_score
df_result["label"] = y_true


#EXTRACTION DEX IMAGES Eronnées

df_erreur = df_result.loc[df_result["label"] != df_result['prédiction']]



#STEP 17 : WANDB summary


wandb.summary['Epoch'] = epoch
wandb.summary['Best_epoch'] = best_epoch
wandb.summary['Validation_loss'] = valid_loss
wandb.summary['accuracy_validation'] = valid_accuracy
wandb.summary['Confusion_matrix_test'] = wandb.Table(columns=columns, data = conf)
wandb.finish()

