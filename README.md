# fruitify
Machine Learning Model to Classiify 131 Classes of Fruits and Vegetables

You can read the approach in [this](https://github.com/ishandutta0098/fruitify/blob/main/fruitify-writeup.pdf) file.

## About the Data
The models are trained on the [Fruits 360 dataset](https://www.kaggle.com/moltean/fruits) from Kaggle.  
`Version: 2020.05.18.0`  

- Total number of images: 90483.
- Training set size: 67692 images (one fruit or vegetable per image).
- Test set size: 22688 images (one fruit or vegetable per image).
- Number of classes: 131 (fruits and vegetables).
- Image size: 100x100 pixels.

## Models
Two models have been implemented.
1. [VGG-16](https://github.com/ishandutta0098/fruitify/blob/main/src/model_vgg16.py)
2. [ResNet-18](https://github.com/ishandutta0098/fruitify/blob/main/src/model_resnet18.py)

![](https://github.com/ishandutta0098/fruitify/blob/main/images/results.png)
