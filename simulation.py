# in this file, we read the data from the path and return as we are acquiring the image from the camera

import cv2
import numpy as np
import os

PATH = './sample_inputs'

def get_fov_id(path):
    # go though the bmp files in the path

    files = os.listdir(path)
    fov_id = []
    for file in files:
        if file.endswith('.bmp'):
            # split with "_" and take the first three, then assemble them together with "_"
            fov_id.append('_'.join(file.split('_')[:3]))
    # Get the unique FOV IDs
    unique_fov_ids = list(set(fov_id))
    
    # Calculate how many times we need to repeat the FOVs to reach 50
    repeat_count = (50 + len(unique_fov_ids) - 1) // len(unique_fov_ids)
    
    # Repeat the FOV IDs to reach at least 50
    extended_fov_ids = unique_fov_ids * repeat_count
    
    # Trim to exactly 50 FOVs if we've exceeded
    extended_fov_ids = extended_fov_ids[:50]
    
    return extended_fov_ids

# now given the list of fov, create a iterator to read the images
def get_image():

    fov_id = get_fov_id(PATH)

    j = 0

    for fov in fov_id:
        # yield left_half, right half, and floresence image sequentially
        yield str(j)

        j += 1

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




'''
def ui_process(input_queue: mp.Queue, output: mp.Queue):
    while True:
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "UI Process", "start")
            
            with final_lock:
                if fov_id in shared_memory_final and not shared_memory_final[fov_id]['displayed']:
                    # Placeholder for UI update
                    temp_dict = shared_memory_final[fov_id]
                    temp_dict['displayed'] = True
                    shared_memory_final[fov_id] = temp_dict

                    if shared_memory_final[fov_id]['saved']:
                        output.put(fov_id)
            
                    log_time(fov_id, "UI Process", "end")
        except Empty:
            continue
'''