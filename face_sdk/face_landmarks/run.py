import os
from utils import *
import time
from utils import *
import torch
from PIL import Image, ImageDraw
img = read_image('output/face.png')
print(img.shape)
img = Image.fromarray(img_to_numpy(img))
xs, ys = get_landmarks('output/template_landmark.txt')
# get all zip files
zip_files = os.listdir('./')
zip_files = [zip_file for zip_file in zip_files if zip_file.endswith('.zip')]

for zip_file in zip_files:
    cmd = 'unzip -o {}'.format(zip_file)
    os.system(cmd)
    for i in range(16):
        cmd = 'python landmarks.py --image_path face_landmarks/origin_{}.png --heatmap_path face_landmarks/heatmap_{}.png'.format(i, i)
        os.system(cmd)

# for i in range(18):
#     cmd = 'python landmarks.py --image_path face_landmarks/origin_{}.png --heatmap_path face_landmarks/heatmap_{}.png'.format(i, i)
#     os.system(cmd)

# cmd = 'python landmarks.py --image_path face_landmarks/origin_{}.png --heatmap_path face_landmarks/heatmap_{}.png'.format(7, 7)
# os.system(cmd)
        
colors = read_colors('/home/yanchen/FaceX-Zoo/face_sdk/colors.txt')
draw = ImageDraw.Draw(img)
colors = cal_avg_colors(colors, 16 * len(zip_files))
# colors = cal_avg_colors(colors, 18)
for x, y, color in zip(xs, ys, colors):
    draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
img.save('output.jpg')
