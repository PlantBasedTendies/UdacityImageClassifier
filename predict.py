'''
    A Computer Vision AI system for predicting flower types.
    Programmer: Michael Gaine
    Date: 07-07-2021

    Sample Command Line (PC):
    python predict.py input_flower.jpg ./checkpoint.pth
'''

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import json
import numpy as np
import PIL
from PIL import Image

# Grab arguments (input, checkpoint, --top_k #, --category_names filename.json, --gpu enabled/disabled)
parser = argparse.ArgumentParser(description='Input arguments')
parser.add_argument('input', type=str, help='flower image relative pathname')
parser.add_argument('checkpoint', type=str, help='checkpoint.pth relative pathname')
parser.add_argument('--top_k', type=int, default=5, help='top K most likely classes')
parser.add_argument('--category_names', type=str, help='category name file (i.e. cat_to_name.json)')
parser.add_argument('--gpu', action="store_true", help="include '--gpu' to use GPU, otherwise CPU will be used")


# Load the command line params and data
args = parser.parse_args()

# Load command line params
if args.top_k:
    top_k = args.top_k
if args.category_names:
    category_names = args.category_names
else:
    category_names = 'cat_to_name.json'
if args.gpu == True:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

image_path = args.input
checkpoint = args.checkpoint


with open(category_names, 'r') as f:
    cat_to_name = json.load(f)


# Load a checkpoint and rebuild the model

def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.hidden_units = checkpoint['hidden_units']
    model.learning_rate = checkpoint['learning_rate']
    model.class_to_idx = checkpoint['class_to_idx']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(model.classifier.parameters(), lr=model.learning_rate)
    return optimizer, model

optimizer, model = load_checkpoint(checkpoint)

# Use appropriate processing unit
model.to(device)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process PIL image for use in a PyTorch model

    pil_image = PIL.Image.open(image_path)

    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    pil_image = img_transform(pil_image)
    global np_image
    np_image = np.array(pil_image)
    return np_image


# Class prediction
def predict(image_path, model, topk = top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Open and process the image
    image = process_image(image_path)

    # Turn on eval mode
    model.eval()

    # Convert numpy array to pytorch tensor and add [0] dimension
    torch_image = torch.tensor(image)
    torch_image = torch_image.unsqueeze(0)

    # Move tensor to GPU
    torch_image = torch_image.to(device)

    # Forward pass
    with torch.no_grad():
        output = model.forward(torch_image)

    # Convert the model output to logits
    ps = torch.exp(output).data

    # Move tensor to CPU
    ps = ps.cpu()

    # Top K
    ps_topk = ps.topk(topk)[0].numpy()[0]
    ps_topk_idx = ps.topk(topk)[1].numpy()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in ps_topk_idx]

    # Return top_prob and top_k variables

    return ps_topk, classes

    print(ps_topk)
    print(classes)



# Output the flower prediction values

ps_topk, classes = predict(image_path, model)
names = [cat_to_name[str(index)] for index in classes]
print('\nThe system has predicted that the flower entered is a: ')
print(names[0].title())
#print(ps_topk)
print('\nWith a probability of:')
print(ps_topk[0])
print('\nThe Top ' + str(top_k) + ' closest types of flower were:')
print(names)