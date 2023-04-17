import os

for i in range(16):
    cmd = 'python landmarks.py --image_path face_landmarks/origin_{}.png --heatmap_path face_landmarks/heatmap_{}.png'.format(i, i)
    os.system(cmd)