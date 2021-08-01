# -*- coding: utf-8 -*-
"""simpleRNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wwgh26pjLKXX4bYAAM16CqxjnK2rfxtw
"""

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transform
import torchvision
import torch
from torch.utils.data import DataLoader
import torch.optim as optimizer
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initializing the parameters
learning_rate=0.001
batch_size=64
num_epochs=10
input_size=28
num_layers=2
hidden_size=256
num_classes=10
sequence_length=28

# creating the RNN
class RNN(nn.Module):
  def __init__(self,hidden_size,num_classes,num_layers,input_size):
    super(RNN,self).__init__()
    self.hidden_size=hidden_size
    self.num_layers=num_layers
    self.rnn=nn.RNN(input_size, hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size*sequence_length,num_classes)
  def forward(self,x):
    h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)

    # forward propagation
    self.rnn(x,h0)
    out,_ =self.rnn(x,h0)
    out=out.reshape(out.shape[0],-1)
    out=self.fc(out)
    return out


# loading the data
train_dataset=datasets.MNIST(root='dataset/',train=True,
                             transform=transform.ToTensor(),download=True)
test_dataset=datasets.MNIST(root='dataset/',train=False,
                            transform=transform.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

# Initializing the network
model=RNN(input_size=input_size,num_classes=num_classes,num_layers=num_layers,
          hidden_size=hidden_size).to(device)
# loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optimizer.Adam(model.parameters(),lr=learning_rate)
for epoch in range(num_epochs):
  for batch_idx,(data,targets) in enumerate(train_loader):
    data=data.to(device).squeeze(1)
    targets=targets.to(device)
    
    #forward
    scores=model(data)
    loss=criterion(scores,targets)
    # backward
    optimizer.zero_grad()
    loss.backward()

    # gradient descent or adam step
    optimizer.step()


def check_accuracy(loader,model):
  if loader.dataset.train:
    print("checking the accuracy on the training data")
  else:
    print("Checking the acuracty on test data")
  num_correct=0
  num_samples=0
  with torch.no_grad():
    for x,y in loader:
      x=x.to(device).squeeze(1)
      y=y.to(device)

      scores=model(x)
      _,predictions=scores.max(1)
      num_correct+=(predictions==y).sum()
      num_samples+=predictions.size(0)

    print(f"Got {num_correct}/{num_samples} with accuracy \
          {float(num_correct)/float(num_samples)*100:.2f}")
  model.train()
check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
