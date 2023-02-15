import os
import random
import subprocess

# get current folder path
current_path = os.getcwd()
# print(current_path)
# data_storage_path= os.path.join(current_path, 'Data/data')
data_storage_path='/home/yanchen/Data/CelebAMask-HQ'
# data_landmark_path=os.path.join(current_path, 'Data/data_landmark')
data_landmark_path='/home/yanchen/Data/data_landmark'
# data_detect_path=os.path.join(current_path, 'Data/data_detect')
data_detect_path='/home/yanchen/Data/data_detect'
# data_mask_path=os.path.join(current_path, 'Data/data_mask')
data_mask_path='/home/yanchen/Data/data_mask'
data_mask_matrix_path='/home/yanchen/Data/data_mask_matrix'

# read all .jpg files in the data_storage_path
file_list = os.listdir(data_storage_path)

# only keep the .jpg files
file_list = [file for file in file_list if file.endswith('.jpg')]
print(len(file_list))

# define a function that append file_name to the end of each line in the .txt file
def append_processed_name(file_name, file_path='./processed_files.txt'):
    file_object = open(file_path, 'a')
    file_object.write(file_name + '\n')

# define a function that check if the file_name is in the processed_files.txt
def check_processed_name(file_name, file_path='./processed_files.txt'):
    file_object = open(file_path)
    try:
        all_the_text = file_object.read()
        if file_name in all_the_text:
            return True
        else:
            return False
    finally:
        file_object.close()

# for each .jpg file, run the command to get the landmark
for file in file_list:
    file_name = file.split('.')[0]
    if check_processed_name(file_name):
        print('already processed ' + file)
        continue
    # generate a random number from 0 - 7, include 0 and 7
    template_name = str(random.randint(0, 7)) + '.png'
    # cd to face_sdk folder
    os.chdir('../../../face_sdk')
    print('start processing ' + file)
    command = 'python api_usage/face_detect.py --image_path ' + data_storage_path + '/' + file + ' --output_path_txt ' + data_detect_path + '/' + file_name + '_detect.txt'
    print(command)
    subprocess.call(command, shell=True)
    command = 'python api_usage/face_alignment.py --image_path ' + data_storage_path + '/' + file + ' --image_det_txt_path ' + data_detect_path + '/' + file_name + '_detect.txt' + ' --output_path_txt ' + data_landmark_path + '/' + file_name + '_landmark.txt'
    print(command)
    subprocess.call(command, shell=True)
    # cd back to the original folder
    os.chdir(current_path)
    command = 'python add_mask_one.py --image_path ' + data_storage_path + '/' + file + ' --face_lms_file ' + data_landmark_path + '/' + file_name + '_landmark0.txt' + ' --masked_face_path ' + data_mask_path + '/' + file_name + '_mask.jpg' + ' --template_name ' + template_name + ' --mask_matrix_path ' + data_mask_matrix_path + '/' + file_name + '_mask_matrix.txt'
    print(command)
    subprocess.call(command, shell=True)
    append_processed_name(file_name)
    print('finish processing ' + file)

