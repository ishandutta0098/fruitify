import config 

import torch 
import torch.nn as nn
import torchvision.models as models 

model = models.vgg16_bn(pretrained = True)
model = model.to(config.DEVICE)