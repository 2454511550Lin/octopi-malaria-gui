# in this file, we read the data from the path and return as we are acquiring the image from the camera

import cv2
import numpy as np
import os

PATH = 'data/pat'

def get_fov_id(path):
    # go though the bmp files in the path

    files = os.listdir(path)
    fov_id = []
    for file in files:
        if file.endswith('.bmp'):
            # split with "_" and take the first three, then assemble them together with "_"
            fov_id.append('_'.join(file.split('_')[:3]))
    
    return list(set(fov_id))

# now given the list of fov, create a iterator to read the images
def get_image():
    fov_id = get_fov_id(PATH)
    for fov in fov_id:
        # yield left_half, right half, and floresence image sequentially
        yield fov

        left_half = cv2.imread(os.path.join(PATH, fov + '_BF_LED_matrix_left_half.bmp'))  

        yield left_half

        right_half = cv2.imread(os.path.join(PATH, fov + '_BF_LED_matrix_right_half.bmp'))

        yield right_half

        floresence = cv2.imread(os.path.join(PATH, fov + '_Fluorescence_405_nm_Ex.bmp'))
    
        yield floresence

# crop the image from 3000x3000 to 2800x2800
def crop_image(image):
    parameters = {}
    parameters['crop_x0'] = 100
    parameters['crop_x1'] = 2900
    parameters['crop_y0'] = 100
    parameters['crop_y1'] = 2900

    return image[parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]