import config 

import torch 
import torch.nn as nn
import torchvision.models as models 

model = models.resnet18(pretrained = True)
model = model.to(config.DEVICE)

