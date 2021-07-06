# Imports here

'''Change batch sizes back to 32 for memory purposes'''

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import json
import os


parser = argparse.ArgumentParser(description='Input arguments')
parser.add_argument('data_dir', type=str, default='flowers', help='data directory')
parser.add_argument('--save_dir', type=str, help='checkpoint directory')
parser.add_argument('--arch', type=str, default='vgg16', help='network architecture')
parser.add_argument('--hidden_units', type=int, default=256, help='hidden units')
parser.add_argument('--learning_rate', type=float, default=.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=1, help='epochs')
parser.add_argument('--gpu', default=True, help='use GPU (True) or CPU (False)')


# Load the command line params and data
args = parser.parse_args()

# Load command line params
if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)


# Map the labels to categories
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# TODO: Build and train your network

# Use GPU if it's available
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load model using architect argument
model = getattr(models, args.arch)(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model.to(device);


## Track loss and accuracy on validation set

#remove this line after testing checkpoint.pth
#criterion = nn.NLLLoss() 

steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        loss = criterion(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    batch_loss = criterion(output, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()

            print(f"Epoch [{epoch+1}/{epochs}]: "
                  f"Train Loss: {running_loss/print_every:.3f} :: "
                  f"Validation Loss: {valid_loss/len(validloader):.3f} :: "
                  f"Validation Accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()


## Track loss and accuracy on the Test set

'''
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        loss = criterion(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    batch_loss = criterion(output, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()

            print(f"Epoch [{epoch+1}/{epochs}]: "
                  f"Train Loss: {running_loss/print_every:.3f} :: "
                  f"Test Loss: {test_loss/len(testloader):.3f} :: "
                  f"Test Accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
'''


# TODO: Save the checkpoint

checkpoint = {'classifier': model.classifier,
              'arch': arch,
              'hidden_units': hidden_units,
              'learning_rate' : learning_rate,
              'epochs': epochs,
              'class_to_idx': train_dataset.class_to_idx, #saves mapping of classes to indices
              'state_dict': model.state_dict(), #saves the learnable parameter layers into a python dictionary
              'optimizer_dict': optimizer.state_dict()}


#torch.save(checkpoint, 'checkpoint.pth')

if args.save_dir:
    os.mkdir(args.save_dir)
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save(checkpoint, 'checkpoint.pth')
