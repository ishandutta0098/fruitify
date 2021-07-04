import config 
import dataset 
import utils 
from model_vgg16 import model 

import torch 
import torch.nn as nn
from tqdm import tqdm
import time
import joblib

utils.seed_everything(config.SEED)

def run():

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        for i, (images, labels) in enumerate(tqdm(dataset.train_dataloader)):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}      Loss: {loss.item()}")
        
    joblib.dump(model, config.MODEL_PATH)  

if __name__ == "__main__":
    run()                  