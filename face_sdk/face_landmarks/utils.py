import torch
import sys
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
    image = cv2.resize(image, (224, 224))
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

def get_std_landmarks():
    normalized_landmarks = [
        (0.233122, 0.160055), (0.254159, 0.248893), (0.284775, 0.334559), (0.319636, 0.415839),
        (0.354911, 0.491390), (0.392334, 0.558926), (0.429429, 0.617122), (0.466262, 0.666746),
        (0.504442, 0.706074), (0.544477, 0.734525), (0.585978, 0.750863), (0.629290, 0.754418),
        (0.670685, 0.745223), (0.706721, 0.720698), (0.737645, 0.684476), (0.764038, 0.639712),
        (0.786889, 0.586892), (0.805464, 0.528317), (0.821318, 0.465296), (0.833384, 0.398379),
        (0.842929, 0.329090), (0.253881, 0.105165), (0.281615, 0.079201), (0.315796, 0.067706),
        (0.351657, 0.069395), (0.385432, 0.084568), (0.621410, 0.083382), (0.655325, 0.067225),
        (0.690735, 0.064993), (0.724386, 0.078356), (0.752264, 0.103659), (0.454566, 0.168815),
        (0.455065, 0.216256), (0.455564, 0.263697), (0.455401, 0.311139), (0.417963, 0.338260),
        (0.439894, 0.349165), (0.463647, 0.356797), (0.486994, 0.349165), (0.508925, 0.338260),
        (0.331091, 0.213413), (0.352815, 0.203874), (0.377697, 0.203874), (0.399421, 0.213413),
        (0.377697, 0.227526), (0.352815, 0.227526), (0.603775, 0.213413), (0.625499, 0.203874),
        (0.650381, 0.203874), (0.672105, 0.213413), (0.650381, 0.227526), (0.625499, 0.227526),
        (0.354911, 0.467890), (0.386436, 0.456422), (0.419271, 0.448467), (0.452106, 0.446575),
        (0.484941, 0.447466), (0.518560, 0.456422), (0.551338, 0.467890), (0.518923, 0.498273),
        (0.486089, 0.523677), (0.453255, 0.530631), (0.420421, 0.523677), (0.387587, 0.498273),
        (0.368908, 0.467890), (0.420421, 0.472141), (0.452106, 0.473032), (0.483791, 0.472141),
        (0.536304, 0.467890), (0.483791, 0.480747), (0.452106, 0.481638), (0.420421, 0.480747),
        (0.206594, 0.159491), (0.227393, 0.136078), (0.254159, 0.131827), (0.280926, 0.136078),
        (0.297891, 0.160055), (0.280926, 0.179095), (0.254159, 0.183346), (0.227393, 0.179095),
        (0.782405, 0.159491), (0.763607, 0.136078), (0.736840, 0.131827), (0.710074, 0.136078),
        (0.691275, 0.160055), (0.710074, 0.179095), (0.736840, 0.183346), (0.763607, 0.179095),
        (0.307949, 0.225946), (0.331091, 0.213413), (0.354911, 0.213413), (0.377697, 0.225946),
        (0.354911, 0.234683), (0.331091, 0.234683), (0.625499, 0.225946), (0.648286, 0.213413),
        (0.672105, 0.213413), (0.695247, 0.225946), (0.672105, 0.234683), (0.648286, 0.234683)
    ]
    return normalized_landmarks
