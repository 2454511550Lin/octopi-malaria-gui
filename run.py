import multiprocessing as mp
#import torch.multiprocessing as mp

from multiprocessing import Lock
import numpy as np
from queue import Empty
from typing import Dict
import time
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from simulation import get_image
import cupy as cp

from log import report
#import torch
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Existing shared memory managers
manager = mp.Manager()
shared_memory_acquisition = manager.dict()
shared_memory_dpc = manager.dict()
shared_memory_segmentation = manager.dict()
shared_memory_fluorescent = manager.dict()
shared_memory_classification = manager.dict()
shared_memory_final = manager.dict()

# New shared memory for timing information
shared_memory_timing = manager.dict()

# Existing locks
dpc_lock = manager.Lock()
segmentation_lock = manager.Lock()
fluorescent_lock = manager.Lock()
classification_lock = manager.Lock()
final_lock = manager.Lock()

# New lock for timing information
timing_lock = manager.Lock()

timeout = 0.1

def log_time(fov_id: str, process_name: str, event: str):
    with timing_lock:
        if fov_id not in shared_memory_timing:
            shared_memory_timing[fov_id] = {}
        if process_name not in shared_memory_timing[fov_id]:
            shared_memory_timing[fov_id][process_name] = {}

        temp_dict = shared_memory_timing[fov_id]
        temp_process_dict = temp_dict.get(process_name, {})
        temp_process_dict[event] = time.time()
        temp_dict[process_name] = temp_process_dict
        shared_memory_timing[fov_id] = temp_dict

from simulation import crop_image

def image_acquisition(dpc_queue: mp.Queue, fluorescent_queue: mp.Queue):

    image_iterator = get_image()

    while True:
        
        # construct the iterator
        try:
            fov_id = next(image_iterator)
            log_time(fov_id, "Image Acquisition", "start")

            left_half = next(image_iterator)
            right_half = next(image_iterator)
            fluorescent = next(image_iterator)

            shared_memory_acquisition[fov_id] = {
                'left_half': crop_image(left_half),
                'right_half': crop_image(right_half),
                'fluorescent': crop_image(fluorescent)
            }

            dpc_queue.put(fov_id)
            fluorescent_queue.put(fov_id)

            log_time(fov_id, "Image Acquisition", "end")
        
        except StopIteration:
            print("No more images to process")
            break
        
        #print(f"Image Acquisition: Processed FOV {fov_id}")
        time.sleep(1) 
         

from utils import generate_dpc, save_dpc_image,save_flourescence_image

def dpc_process(input_queue: mp.Queue, output_queue: mp.Queue):
    while True:
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "DPC Process", "start")
            
            data = shared_memory_acquisition[fov_id]
            left_half = data['left_half'].astype(float)/255
            right_half = data['right_half'].astype(float)/255
            
            dpc_image = generate_dpc(left_half, right_half,use_gpu=False)
            
            with dpc_lock:
                shared_memory_dpc[fov_id] = {'dpc_image': dpc_image}
            
            output_queue.put(fov_id)
            log_time(fov_id, "DPC Process", "end")
        except Empty:
            continue

def segmentation_process(input_queue: mp.Queue, output_queue: mp.Queue):
    while True:
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "Segmentation Process", "start")
            
            # Wait for DPC to finish
            while fov_id not in shared_memory_dpc:
                time.sleep(timeout)
            
            dpc_image = shared_memory_dpc[fov_id]['dpc_image']
            
            # Placeholder for segmentation
            segmentation_map = dpc_image > 0.5  # This is just a placeholder operation
            #print(f"Segmentation Process: Processed FOV {fov_id}")
            
            with segmentation_lock:
                shared_memory_segmentation[fov_id] = {'segmentation_map': segmentation_map}
            
            output_queue.put(fov_id)
            log_time(fov_id, "Segmentation Process", "end")
        except Empty:
            continue

from utils import remove_background, resize_image_cp, detect_spots, prune_blobs, settings
import cProfile

def fluorescent_spot_detection(input_queue: mp.Queue, output_queue: mp.Queue):
    
    while True:
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "Fluorescent Spot Detection", "start")
            
            fluorescent = shared_memory_acquisition[fov_id]['fluorescent']

            I_fluorescence_bg_removed = remove_background(fluorescent,return_gpu_image=True)

            # detect spots
            log_time(fov_id,"detect spots","start")
            spot_list = detect_spots(resize_image_cp(I_fluorescence_bg_removed,
                                                     downsize_factor=settings['spot_detection_downsize_factor']),
                                                     thresh=settings['spot_detection_threshold'])
            log_time(fov_id,"detect spots","end")

            log_time(fov_id,"prune blobs","start")
            spot_list = prune_blobs(spot_list)
            log_time(fov_id,"prune blobs","end")

            # scale coordinates for full-res image
            spot_list = spot_list*settings['spot_detection_downsize_factor']
            
            with fluorescent_lock:
                shared_memory_fluorescent[fov_id] = {'spot_indices': spot_list}
            
            output_queue.put(fov_id)
            log_time(fov_id, "Fluorescent Spot Detection", "end")

        except Empty:
            continue

from utils import get_spot_images_from_fov

def classification_process(segmentation_queue: mp.Queue, fluorescent_queue: mp.Queue, save_queue: mp.Queue, ui_queue: mp.Queue):
    segmentation_ready = set()
    fluorescent_ready = set()

    # initalize model
    #CHECKPOINT = './checkpoint/resnet18_en/version1/best.pt'
    #model = ResNet('resnet18').to(device=DEVICE)
    #model.load_state_dict(torch.load(CHECKPOINT))
    #model.eval()

    while True:
        try:
            # Check segmentation queue
            try:
                fov_id = segmentation_queue.get(timeout=timeout)
                segmentation_ready.add(fov_id)
            except Empty:
                pass

            # Check fluorescent queue
            try:
                fov_id = fluorescent_queue.get(timeout=timeout)
                fluorescent_ready.add(fov_id)
            except Empty:
                pass

            # Process FOVs that are ready in both queues
            ready_fovs = segmentation_ready.intersection(fluorescent_ready)
            for fov_id in ready_fovs:
                log_time(fov_id, "Classification Process", "start")
                print(f"Classification Process: Processing FOV {fov_id}")
     
                segmentation_map = shared_memory_segmentation[fov_id]['segmentation_map']
                spot_list = shared_memory_fluorescent[fov_id]['spot_indices']
                
                dpc_image = shared_memory_dpc[fov_id]['dpc_image']
                fluorescence_image = shared_memory_acquisition[fov_id]['fluorescent'].astype(float)/255
                
                print(f"Classification Process: getting spot images for FOV {fov_id} with {len(spot_list)} spots")
                cropped_images = get_spot_images_from_fov(fluorescence_image,dpc_image,spot_list,r=15)
                cropped_images = cropped_images.transpose(0, 3, 1, 2)

                print(f"Classification Process: got spot images for FOV {fov_id}")
                #scores = run_model(model,DEVICE,cropped_images,4096)[:,1]
                # random scores
                scores = np.random.rand(len(spot_list))

                with classification_lock:
                    shared_memory_classification[fov_id] = {
                        'cropped_images': cropped_images,
                        'scores': scores
                    }

                print(f"Classification Process: Processed FOV {fov_id}")
                
                # Update shared_memory_final
                with final_lock:
                    if fov_id not in shared_memory_final:
                        shared_memory_final[fov_id] = {
                            'saved': False,
                            'displayed': False
                        }

                save_queue.put(fov_id)
                ui_queue.put(fov_id)
                segmentation_ready.remove(fov_id)
                fluorescent_ready.remove(fov_id)

                #print(f"Classification Process: Processed FOV {fov_id}")
                log_time(fov_id, "Classification Process", "end")

        except Exception as e:
            print(f"Error in classification process: {e}")
            continue

from utils import numpy2png
import os

def saving_process(input_queue: mp.Queue, output: mp.Queue):
    while True:
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "Saving Process", "start")
            
            with final_lock:
                if fov_id in shared_memory_final and not shared_memory_final[fov_id]['saved']:
                    
                    # save the cropped images to png
                    cropped_images = shared_memory_classification[fov_id]['cropped_images']*255
                    print(f"Saving Process: Saving {len(cropped_images)} images for FOV {fov_id} with shape {cropped_images[0].shape}")

                    path = 'cropped_images'
                    # randomly select 10 images
                    random_indices = np.random.choice(len(cropped_images),1,replace=False)
                    for i, cropped_image in enumerate(cropped_images[random_indices]):
                        filename = os.path.join(path, f"{fov_id}_{i}.png")
                        numpy2png(cropped_image,filename)

                    #dpc_image = shared_memory_dpc[fov_id]['dpc_image']*255
                    #save_dpc_image(dpc_image, f'data/{fov_id}.png')
                      
                    temp_dict = shared_memory_final[fov_id]
                    temp_dict['saved'] = True
                    shared_memory_final[fov_id] = temp_dict

                    if shared_memory_final[fov_id]['displayed']:
                        #print(f"Saving Process: FOV {fov_id} is ready for cleanup")
                        output.put(fov_id)
            
                    log_time(fov_id, "Saving Process", "end")
        except Empty:
            continue

def ui_process(input_queue: mp.Queue, output: mp.Queue):
    while True:
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "UI Process", "start")
            
            with final_lock:
                if fov_id in shared_memory_final and not shared_memory_final[fov_id]['displayed']:
                    # Placeholder for UI update
                    #print(f"UI Process: Updated UI for FOV {fov_id}")
                    temp_dict = shared_memory_final[fov_id]
                    temp_dict['displayed'] = True
                    shared_memory_final[fov_id] = temp_dict

                    if shared_memory_final[fov_id]['saved']:
                        #print(f"UI Process: FOV {fov_id} is ready for cleanup")
                        output.put(fov_id)
            
                    log_time(fov_id, "UI Process", "end")
        except Empty:
            continue

        

def cleanup_process(cleanup_queue: mp.Queue):
    while True:
        try:
            fov_id = cleanup_queue.get(timeout=timeout)
            
            # Clean up shared memory
            with final_lock, timing_lock:

                # Calculate processing times and generate visualization
                timing_data = shared_memory_timing[fov_id]
                total_time = timing_data['UI Process']['end'] - timing_data['Image Acquisition']['start']

                report(timing_data, fov_id)

                for shared_memory in [shared_memory_acquisition, shared_memory_dpc, shared_memory_segmentation, shared_memory_fluorescent, shared_memory_classification, shared_memory_final, shared_memory_timing]:
                    if fov_id in shared_memory:
                        del shared_memory[fov_id]
            print(f"FOV {fov_id} memory is freed up")
        except Empty:
            continue

from model import ResNet, run_model

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')  # For CUDA compatibility with multiprocessing
    except RuntimeError:
        pass  # Ignore if the context has already been set
    # Create queues
    dpc_queue = mp.Queue()
    fluorescent_queue = mp.Queue()
    segmentation_queue = mp.Queue()
    fluorescent_detection_queue = mp.Queue()
    classification_queue = mp.Queue()
    save_queue = mp.Queue()
    ui_queue = mp.Queue()
    cleanup_queue = mp.Queue()

    print("Main")

    # Create and start processes
    processes = [
        mp.Process(target=image_acquisition, args=(dpc_queue, fluorescent_queue)),
        mp.Process(target=dpc_process, args=(dpc_queue, segmentation_queue)),
        mp.Process(target=segmentation_process, args=(segmentation_queue, classification_queue)),
        mp.Process(target=fluorescent_spot_detection, args=(fluorescent_queue, fluorescent_detection_queue)),
        mp.Process(target=classification_process, args=(classification_queue, fluorescent_detection_queue, save_queue, ui_queue)),
        mp.Process(target=saving_process, args=(save_queue, cleanup_queue)),
        mp.Process(target=ui_process, args=(ui_queue, cleanup_queue)),
        mp.Process(target=cleanup_process, args=(cleanup_queue,)),
    ]

    for p in processes:
        p.start()

    try:
        # Wait for all processes to complete
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping all processes...")
        for p in processes:
            p.terminate()