import config 
import dataset 

import joblib
import torch 

model = joblib.load(config.MODEL_PATH)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for (images, labels) in dataset.test_dataloader:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        outputs = model(images)
        _, predict = torch.max(outputs.data, 1)
        total+=labels.size(0)
        correct+=(predict==labels).sum().item()
    
    print('Test accuracy: {}'.format(100*correct/total))     