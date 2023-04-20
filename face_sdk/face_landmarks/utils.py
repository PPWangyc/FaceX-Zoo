import torch
import sys
import os
sys.path.append('/home/yanchen/facial-expression/fatigue')
from data import FaceF_whole_video_2_labels
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import wandb
from models.ST_Former import GenerateModel, RVT
import matplotlib.pyplot as plt
import cv2
import numpy as np
import __main__
import torchvision.models as models
from visualizer import get_local
get_local.activate()
import torch.nn.functional as F
from captum.attr import LayerGradCam
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap, Normalize

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
def load_net():
    net = GenerateModel()
    net = RVT(net)
    state = torch.load('/home/yanchen/facial-expression/checkpoint/rvt.pth')
    net.load_state_dict(state['net'])
    # net.fc = nn.Linear(512, 2)
    return net.to(device)
hidden = torch.zeros(1, 2).to(device)
net = load_net()

def model_with_hidden(input_tensor):
    output, _ = net(input_tensor, hidden)
    return output

def get_attributions(net, wrap_net,  input_tensor, target):
    target_layer = net.generate_model.s_former.layer1[1].conv2
    gradcam = LayerGradCam(wrap_net, target_layer)
    return gradcam.attribute(input_tensor, target)

# read image from path and convert to tensor
def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (224, 224))
    image = transforms.ToTensor()(image)
    # image = image.unsqueeze(0)
    return image.to(device)

def img_to_numpy(img):
    img = img.detach().cpu().numpy()
    img = img.transpose(1, 2, 0) * 255
    img = img.astype(np.uint8)
    return img

def get_landmarks(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    l = lines[0].strip().split()
    # get all xs and ys
    xs = []
    ys = []
    for i in range(0, len(l), 2):
        xs.append(int(l[i]))
        ys.append(int(l[i+1]))
    return xs, ys

def save_colors(colors):
    with open("colors.txt", "w") as f:
        for color in colors:
            f.write(f"{color[0]} {color[1]} {color[2]}\n")

def cal_avg_colors(colors, devide = 16):
    # every color tuple divide by 16
    return [(x[0] // devide, x[1] // devide, x[2] // devide) for x in colors]
    

def read_colors(path="colors.txt"):
    colors = []
    # if file does not exist, create it
    if not os.path.exists(path):
        color = (0, 0, 0)
        for i in range(106):
            colors.append(color)
        save_colors(colors)
    else:
        with open(path, "r") as f:
            for line in f.readlines():
                l = line.strip().split()
                colors.append((int(l[0]), int(l[1]), int(l[2])))
    return colors

def get_average_color(img, x, y, radius):
    total_pixels = 0
    total_color = np.array([0, 0, 0])
    
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i**2 + j**2 <= radius**2:
                xi, yj = x + i, y + j
                
                if 0 <= xi < img.width and 0 <= yj < img.height:
                    total_color += np.array(img.getpixel((xi, yj)))
                    total_pixels += 1
    
    return tuple((total_color / total_pixels).astype(np.uint8))

def sum_colors(old, new):
    sum_colors = []
    for i in range(len(old)):
        new_tuple = tuple(map(sum, zip(old[i], new[i])))
        sum_colors.append(new_tuple)
    return sum_colors
