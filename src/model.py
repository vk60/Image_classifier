import torch.nn as nn
from torchvision import models

def get_model():
    model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
   
    for param in model.parameters():
        param.requires_grad=False
        # model.fc  shows in_features=512 and out_features=1000
    model.fc=nn.Sequential(
        nn.Linear(model.fc.in_features,128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128,2)
    )
    return model    
