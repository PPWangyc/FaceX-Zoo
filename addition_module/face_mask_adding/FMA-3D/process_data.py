import os
import subprocess

# get current folder path

current_path = os.getcwd()
print(current_path)
data_storage_path= os.path.join(current_path, '/Data/data')
data_landmark_path=os.path.join(current_path, '/Data/data_landmark')
data_detect_path=os.path.join(current_path, '/Data/data_detect')

print(data_storage_path)

# read all .jpg files in the data_storage_path
file_list = os.listdir(data_storage_path)

# only keep the .jpg files
file_list = [file for file in file_list if file.endswith('.jpg')]
print(file_list)

# cd to face_sdk folder
os.chdir('../../../face_sdk')

# list all files in current folder
print(os.listdir())

# for each .jpg file, run the command to get the landmark
for file in file_list:
    file_name = file.split('.')[0]
    command = 'python api_usage/face_detect.py --image_path ' + data_storage_path + '/' + file + ' --output_path_txt ' + data_detect_path + '/' + file_name + '_detect.txt'
    print(command)
    subprocess.call(command, shell=True)
    command = 'python api_usage/face_alignment.py --image_path ' + data_storage_path + '/' + file + ' --image_det_txt_path ' + data_detect_path + '/' + file_name + '_detect.txt' + ' --output_path_txt ' + data_landmark_path + '/' + file_name + '_landmark.txt'
    print(command)
    subprocess.call(command, shell=True)