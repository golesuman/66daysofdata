# -*- coding: utf-8 -*-
"""leafDiesease.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qbLc-TmR_r0BV0DhmozANUPajIQmhqWG
"""

! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download -d tahsin/cassava-leaf-disease-merged

! unzip cassava-leaf-disease-merged.zip

import pandas as pd
import os
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

df=pd.read_csv('merged.csv')
df.head()

df.label.value_counts()

image_path='/content/train'
train_paths=[os.path.join(image_path,x) for x in df.image_id.values]
train_labels=[x for x in df.label.values]
print(train_labels[:5])
print(train_paths[:5])
len(train_paths)
len(train_labels)

from skimage import io
import cv2
import torch

import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset,DataLoader
class leafDisease(Dataset):
  def __init__(self,csv_file,root_dir,transform):
    self.annotations=pd.read_csv(csv_file)
    self.root_dir=root_dir
    self.transform=transform
  

  def __len__(self):
    return len(self.annotations)


  def __getitem__(self,index):
    img_path=os.path.join(self.root_dir,self.annotations.iloc[index,0])
    image=io.imread(img_path)
    y_label=torch.tensor(int(self.annotations.iloc[index,1]))

    if self.transform:
      image=self.transform(image)
      return (image,y_label)

class ImageDataset:
    def __init__(
        self,
        image_paths,
        targets,
        augmentations=None,
        backend="pil",
        channel_first=True,
        grayscale=False,
    ):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param augmentations: albumentations augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.backend = backend
        self.channel_first = channel_first
        self.grayscale = grayscale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        targets = self.targets[item]
        if self.backend == "pil":
            image = Image.open(self.image_paths[item])
            image = np.array(image)
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
        elif self.backend == "cv2":
            if self.grayscale is False:
                image = cv2.imread(self.image_paths[item])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.imread(self.image_paths[item], cv2.IMREAD_GRAYSCALE)
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
        else:
            raise Exception("Backend not implemented")
        if self.channel_first is True and self.grayscale is False:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image_tensor = torch.tensor(image)
        if self.grayscale:
            image_tensor = image_tensor.unsqueeze(0)
        return {
            "image": image_tensor,
            "targets": torch.tensor(targets),
        }

train_dataset=ImageDataset(image_paths='/content/train',targets=train_labels)
train_dataset[0]

dataset=leafDisease(csv_file='/content/merged.csv',root_dir='/content/train',transform=transforms.ToTensor())
train_set,test_set=torch.utils.data.random_split(dataset, [20000, 6337])
train_loader=DataLoader(train_set,batch_size=100,shuffle=True)
test_loader=DataLoader(test_set,batch_size=100,shuffle=True)

for samples,labels in train_loader:
  print(labels)
