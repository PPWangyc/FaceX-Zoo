import argparse
import sys
import os
from utils import *
import torch
from PIL import Image, ImageDraw
parser = argparse.ArgumentParser(description='Face Landmarks')
parser.add_argument('--image_path', type=str, default='face_landmarks/origin_1.png', help='path to input image')
parser.add_argument('--heatmap_path', type=str, default='face_landmarks/heatmap_1.png', help='path to input image')
parser.add_argument('--image_det_txt_path', type=str, default='face_landmarks/temp/test1_detect_res.txt', help='path to input detect txt')
parser.add_argument('--output_path', type=str, default='face_landmarks/output/test1_landmark_res.jpg', help='path to save output image')
parser.add_argument('--output_path_txt', type=str, default='face_landmarks/output/test1_landmark_res.txt', help='path to save landmark txt')

args = parser.parse_args()
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
# process input image and get detection results
os.chdir('/home/yanchen/FaceX-Zoo/face_sdk')
# read txt file image_det_txt_path
# get the first line

# cd to face_sdk/face_landmarks
num  = args.image_path.split('/')[-1].split('.')[0].split('_')[-1]
image_det_txt_path = 'api_usage/test_images/test1_detect_res.txt'
# run command like: python ./api_usage/face_detection.py --image_path args.image_path --output_path_txt args.image_det_txt_path
cmd = 'python ./api_usage/face_detect.py --image_path ' + args.image_path + ' --output_path_txt ' + args.image_det_txt_path
os.system(cmd)
cmd = 'python ./api_usage/face_alignment.py --image_path ' + args.image_path + ' --image_det_txt_path ' + args.image_det_txt_path + ' --output_path ' + args.output_path + ' --output_path_txt ' + args.output_path_txt
os.system(cmd)

xs, ys = get_landmarks(args.output_path_txt)
img = read_image(args.image_path)
img = Image.fromarray(img_to_numpy(img))

img_hp = read_image(args.heatmap_path)
img_hp = Image.fromarray(img_to_numpy(img_hp))

draw = ImageDraw.Draw(img)

radius = 1
# def calculate the area value of the heatmap in the xs, ys
# colors = read_colors()
# temp = []
colors = []
for x, y in zip(xs, ys):
    # x = 2*x
    # y = 2*y
    # Get the color from the heatmap at the landmark position
    color = get_average_color(img_hp, x, y, 5)
    # temp.append(color)
    colors.append(color)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

img.save('output_{}.png'.format(num))
# colors = sum_colors(colors, temp)
print(colors[2:5])
save_colors(colors)

def get_colors():
    return colors
