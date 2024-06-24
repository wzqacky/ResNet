from model import *

import torch 
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import autoaugment
from tqdm import tqdm
import matplotlib.pyplot as plt

img_channels = 3
img_classes = 1000
batch_size = 64
torch.cuda.manual_seed(42)

# calculate the accuracy of the model 
def cal_accuracy(loader, model, loss_fn, device):
    correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        validation_loss_per_epoch = []
        for idx, (data, labels) in enumerate(loader):
            data = data.to(device)
            labels = labels.to(device)
            true = labels
            labels = labels.type(torch.long)
            pred = model(data)
            loss = loss_fn(pred, labels)
            validation_loss_per_epoch.append(loss.item())
            _, predictions = pred.max(1)
            correct += (predictions==true).sum()
            num_samples += predictions.size(0)
    print(f"Validation accuracy: {correct/num_samples:>0.3f}")
    return sum(validation_loss_per_epoch)/len(validation_loss_per_epoch)

# data preprocessing 
policy = autoaugment.AutoAugmentPolicy('imagenet')
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224), scale=(0.08, 1.0), ratio=(3/4, 4/3)),
    transforms.RandomHorizontalFlip(p=0.5),
    autoaugment.AutoAugment(policy=policy),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

validation_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

train_set = ImageFolder("data/train", train_transform)
validation_set = ImageFolder("data/val", validation_transform)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True, drop_last=True)

# hyperparameters
lr = 1e-3
num_epochs = 50
model = ResNet([3, 4, 6, 3], img_channels, img_classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

# train the model 
train_loss = []
validation_loss = []
for epoch in tqdm(range(num_epochs)):
    training_loss_per_epoch = []
    model.train()
    for idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        labels = labels.type(torch.long)
        pred = model(data)
        loss = loss_fn(pred, labels)
        training_loss_per_epoch.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    average_loss = sum(training_loss_per_epoch)/len(training_loss_per_epoch)
    print(f"At epoch {epoch+1}, the loss is {average_loss}")
    train_loss.append(average_loss)
    validation = cal_accuracy(validation_loader, model, loss_fn, device)
    scheduler.step(validation)
    validation_loss.append(validation)

# save the model 
torch.save({
            'epoch': num_epochs,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            }, "ResNet50")

# plot the loss 
# plotting the loss 
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot([epoch+1 for epoch in range(num_epochs)], train_loss, 'r', label="Training Loss")
plt.plot([epoch+1 for epoch in range(num_epochs)], validation_loss, 'b', label="Validation Loss")
plt.legend()
plt.show()
