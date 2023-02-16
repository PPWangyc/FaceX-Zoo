from skimage.io import imread, imsave
import numpy as np

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

# define a function that save mask matrix and convert to an image
mask_matrix = get_mask_matrix('/home/yanchen/Data/data_mask_matrix/1251_mask_matrix.txt')
# convert mask_matrix to numpy integer array
mask_matrix = mask_matrix.astype(int)

upper_face_matrix = get_mask_matrix('/home/yanchen/Data/data_upper_face_matrix/1251_upper_face_matrix.txt')
upper_face_matrix = upper_face_matrix.astype(int)
imsave('test.jpg', mask_matrix)
imsave('test2.jpg', upper_face_matrix)
