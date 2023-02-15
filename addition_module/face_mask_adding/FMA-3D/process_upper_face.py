import os
import numpy as np
from skimage.io import imread, imsave

data_storage_path='/home/yanchen/Data/CelebAMask-HQ'
data_detect_path='/home/yanchen/Data/data_detect'
data_mask_matrix_path='/home/yanchen/Data/data_mask_matrix'
data_upper_face_matrix_path='/home/yanchen/Data/data_upper_face_matrix'

# read all .txt files in mask_matrix_path
file_list = os.listdir(data_mask_matrix_path)

# only keep the .txt files
file_list = [file for file in file_list if file.endswith('.txt')]

# define a function to read the mask matrix from the .txt file
def get_mask_matrix(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        mask_matrix = []
        for line in lines:
            line = line.strip()
            line = line.split(' ')
            mask_matrix.append(line)
        mask_matrix = np.array(mask_matrix)
    return mask_matrix

# define a function to read the detect file from the .txt file
def get_detect(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        detect = []
        for line in lines:
            line = line.strip()
            line = line.split(' ')
            detect.append(line)
        detect = np.array(detect)[0]
    return detect

# define a function: if the mask_matrix tuple is 1, then set the corresponding pixel in the image to 0
def add_mask(image, mask_matrix):
    for i in range(mask_matrix.shape[0]):
        for j in range(mask_matrix.shape[1]):
            if mask_matrix[i][j] == '1':
                image[i][j] = 0
    return image

# define a function: if the image pixel is out of the detect bbox, then set the pixel to 0
def add_detect(image, detect_bbox):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i < int(detect_bbox[1]) or i > int(detect_bbox[3]) or j < int(detect_bbox[0]) or j > int(detect_bbox[2]):
                image[i][j] = 0
    return image

# define a function: if the image pixel is not zero, then convert the pixel to 1
def add_upper_face(image):
    image = np.where(image !=0, 1, 0)
    return image

# define a function: convert the image to 2d, and save the image
def save_image(image, file_name):
    image = image[:,:,0]
    np.savetxt(data_upper_face_matrix_path + '/' + file_name + '_upper_face_matrix.txt', image, fmt='%d')
    # imsave(data_upper_face_matrix_path + '/' + file_name + '_upper_face_matrix.jpg', image)

# define a function: append processed filename processed_upper_face.txt
def append_processed_upper_face(file_name, file_path='./processed_upper_face.txt'):
    file_object = open(file_path, 'a')
    file_object.write(file_name + '\n')

# define a function: check if the file has been processed
def check_processed_upper_face(file_name, file_path='./processed_upper_face.txt'):
    file_object = open(file_path)
    try:
        all_the_text = file_object.read()
        if file_name in all_the_text:
            return True
        else:
            return False
    finally:
        file_object.close()

# for each .txt file, run the command to get the mask matrix
for file in file_list:
    if check_processed_upper_face(file):
        print("File: " + file + " has been processed")
        continue
    print("Processing file: " + file)
    file_name = file.split('.')[0].split('_')[0]
    mask_matrix = get_mask_matrix(data_mask_matrix_path + '/' + file)
    # read the image
    image = imread(data_storage_path + '/' + file_name + '.jpg')
    detect_bbox = get_detect(data_detect_path + '/' + file_name + '_detect.txt')
    # add mask
    image = add_mask(image, mask_matrix)
    # add detect
    image = add_detect(image, detect_bbox)
    # add upper face
    image = add_upper_face(image)
    save_image(image, file_name)
    append_processed_upper_face(file)
    print("Done with file: " + file)

    