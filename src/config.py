import torch

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DATA_DIRECTORY = "D:/Kaggle/Fruitify (fruits360)/input"
TRAIN_DATA_PATH = DATA_DIRECTORY + "/Training"
TEST_DATA_PATH = DATA_DIRECTORY + "/Test"
MODEL_PATH = "D:/Kaggle/Fruitify (fruits360)/outputs/resnet18.bin"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 1
SEED = 42
