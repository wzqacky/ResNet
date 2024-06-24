from model import *

import matplotlib.pyplot as plt 
import torchvision.utils as vutils
import numpy as np
import torch 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms


def cal_accuracy(loader, model, device):
    correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for idx, (data, labels) in enumerate(loader):
            data = data.to(device)
            labels = labels.to(device)
            true = labels
            labels = labels.type(torch.long)
            pred = model(data)
            _, predictions = pred.max(1)
            print(f"model prediction: {predictions[0]}")
            print(f"groud truth: {true[0]}")
            correct += (predictions==true).sum()
            num_samples += predictions.size(0)
    print(f"Validation accuracy: {correct/num_samples:>0.3f}")

img_channels = 3
img_classes = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device) 

validation_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

validation_set = ImageFolder("data/val", validation_transform)
validation_loader = DataLoader(dataset=validation_set, batch_size=1, shuffle=True)

# make prediction on one image 
batch = next(iter(validation_loader))
model = ResNet([3, 4, 6, 3], img_channels, img_classes)
loaded = torch.load("ResNet50")
model.load_state_dict(loaded['model'])
model.eval()
pred = model(batch[0])
_, prediction = pred.max(1)

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Test Image")
plt.imshow(np.transpose(vutils.make_grid(batch[0].detach().cpu(), normalize=True), (1, 2, 0)))
print(f"model prediction: {prediction[0]}")
print(f"groud truth: {batch[1][0]}")
plt.show()

# for evaluating model accuracy 
"""
model.to(device)
cal_accuracy(validation_loader, model, device)
"""