# Summary 
The repository serves as an exercise for implementing ResNet in the classification task of human faces 

# Dataset 
The dataset consists of human faces from 1,000 different people 

# train.py
### 1. Data Augmentation 
A few data augmentation tricks are utilized
- Random resized crop
- Random horizontal flip
- AutoAugment using ImageNet policy
- Normalization

### 2. Tuning hyperparameters
The pretrained model (ResNet50) is trained by the following hyperparameters:
- learning rate: 1e-3
- number of epochs: 100
- loss function: Cross Entropy Loss
- optimizer: Adam
- scheduler: ReducedLROnPlateau (Reducing the learning rate when the provided metric stop improving)

### 3. Plotting the loss 
![loss](https://github.com/wzqacky/ResNet/assets/100191968/6c3c2619-59ce-4203-a64d-d64d973e1801)

# model.py
The script of the whole model architecture 

Different scale of ResNet requries different number of layers which could be customized by specifying the layer list for the initialization of ResNet 

For example, ResNet50 consists of 3, 4, 6, 3 residual blocks respectively at four levels, represented by layer=[3,4,6,3]

In addition, some patterns could be found in the following graph
- The ouput size halves each level
- Stride in the first level is 1 while the others are 2
- FLOPs remains about the same when 34-layer is upscaled to 50-layer thanks to the bottleneck design in the residual block

![Res ](https://github.com/wzqacky/ResNet/assets/100191968/fda0d7d7-2b81-4f9b-b5d0-f0f9ad08df01)


# test.py
Using the pretrained model to generate output on one image and evaluate its accuracy 

Validation accuracy: 0.770

