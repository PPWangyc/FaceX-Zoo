"""
@author: Yinglu Liu, Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

# add arguments
import argparse

argparser = argparse.ArgumentParser(description='Face Masking')
argparser.add_argument('--image_path', type=str, default='Data/test-data/test1.jpg', help='path to input image')
argparser.add_argument('--face_lms_file', type=str, default='Data/test-data/test1_landmark.txt', help='path to input image')
argparser.add_argument('--template_name', type=str, default='7.png', help='path to input image')
argparser.add_argument('--masked_face_path', type=str, default='test1_mask1.jpg', help='path to input image')
argparser.add_argument('--mask_matrix_path', type=str, default='test1_mask1_matrix.txt', help='path to input image')

args = argparser.parse_args()

from face_masker import FaceMasker

if __name__ == '__main__':
    is_aug = False
    image_path = args.image_path
    face_lms_file = args.face_lms_file
    template_name = args.template_name
    masked_face_path = args.masked_face_path
    mask_matrix_path = args.mask_matrix_path
    face_lms_str = open(face_lms_file).readline().strip().split(' ')
    face_lms = [float(num) for num in face_lms_str]
    face_masker = FaceMasker(is_aug)
    face_masker.add_mask_one(image_path, face_lms, template_name, masked_face_path, mask_matrix_path)
