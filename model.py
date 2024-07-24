import torch.nn as nn
import torchvision

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import time



class ResNet(nn.Module):
  def __init__(self, model='resnet18', n_channels=4, n_filters=64, n_classes=2, kernel_size=3, stride=1, padding=1):
    super().__init__()
    self.n_classes = n_classes
    models_dict = {
      'resnet18': torchvision.models.resnet18,
      'resnet34': torchvision.models.resnet34,
      'resnet50': torchvision.models.resnet50,
      'resnet101': torchvision.models.resnet101,
      'resnet152': torchvision.models.resnet152
    }
    self.base_model = models_dict[model](weights=None)
    self._feature_vector_dimension = self.base_model.fc.in_features
    self.base_model.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # Remove the final fully connected layer
    self.fc = nn.Linear(self._feature_vector_dimension, n_classes)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.base_model(x)
    features = x.view(x.size(0), -1)
    x = self.fc(features)
    return x, features
  

# runs model
def run_model(model, device, images, batch_size_inference=512):

    return generate_predictions(model,device,images,batch_size_inference)

from torch.nn.functional import softmax

def generate_predictions(model, device, images, batch_size_inference):
    if images.dtype == np.uint8:
        images = images.astype(np.float32)/255.0 # convert to 0-1 if uint8 input

    # build dataset
    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(np.ones(images.shape[0])))

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_inference, shuffle=False)

    # run inference 
    all_predictions = []
    for k, (images, labels) in enumerate(dataloader):
        images = images.float().to(device)
        predictions, _ = model.forward(images)
        predictions = softmax(predictions, dim=1)  # Apply softmax to convert logits to probabilities
        predictions = predictions.detach().cpu().numpy()
        all_predictions.append(predictions)
        del images
        del labels
        #torch.cuda.empty_cache()  # Clear GPU memory
        
    del dataset
    del dataloader
    predictions = np.vstack(all_predictions)
  

    return predictions