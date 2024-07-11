#import multiprocessing as mp
#from multiprocessing import Lock

import torch.multiprocessing as mp
import numpy as np
from queue import Empty
import time

from simulation import get_image
import threading
from log import report

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

# time lock
timing_lock = manager.Lock()

timeout = 0.1


def log_time(fov_id: str, process_name: str, event: str):
    with timing_lock:
        #print(f"Logging time for FOV {fov_id} in {process_name} at {event}")
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

            # left and right should be 2800x2800, if three channels, convert to grayscale
            if left_half.shape[2] == 3:
                left_half = left_half[:,:,0]
            if right_half.shape[2] == 3:
                right_half = right_half[:,:,0]
    
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
        time.sleep(0.5) 
         

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

            try:
                assert dpc_image.shape == (2800, 2800)
            except AssertionError:
                print(f"FOV {fov_id} DPC image shape is {dpc_image.shape} but not (2800, 2800)")
            
            with dpc_lock:
                shared_memory_dpc[fov_id] = {'dpc_image': dpc_image}
            
            output_queue.put(fov_id)
            log_time(fov_id, "DPC Process", "end")
        except Empty:
            continue


def segmentation_process(input_queue: mp.Queue, output_queue: mp.Queue):
    from interactive_m2unet_inference import M2UnetInteractiveModel as m2u
    import torch
    from scipy.ndimage import label

    PATH = 'checkpoint/m2unet_model_flat_erode1_wdecay5_smallbatch/model_4000_11.pth'
    model = m2u(pretrained_model=PATH, use_trt=False)
    
    while True:
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "Segmentation Process", "start")
            
            # Wait for DPC to finish
            while fov_id not in shared_memory_dpc:
                time.sleep(timeout)
            
            dpc_image = shared_memory_dpc[fov_id]['dpc_image']
            
            result = model.predict_on_images(dpc_image)
            threshold = 0.5
            segmentation_mask = (255*(result > threshold)).astype(np.uint8)
            _, n_cells = label(segmentation_mask)
            
            with segmentation_lock:
                shared_memory_segmentation[fov_id] = {'segmentation_map': segmentation_mask, 'n_cells': n_cells}
            
            output_queue.put(fov_id)
            log_time(fov_id, "Segmentation Process", "end")
        except Empty:
            continue

from utils import remove_background, resize_image_cp, detect_spots, prune_blobs, settings

def fluorescent_spot_detection(input_queue: mp.Queue, output_queue: mp.Queue):
    
    while True:
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "Fluorescent Spot Detection", "start")
            
            fluorescent = shared_memory_acquisition[fov_id]['fluorescent']

            I_fluorescence_bg_removed = remove_background(fluorescent,return_gpu_image=True)

            spot_list = detect_spots(resize_image_cp(I_fluorescence_bg_removed,
                                                     downsize_factor=settings['spot_detection_downsize_factor']),
                                                     thresh=settings['spot_detection_threshold'])

            spot_list = prune_blobs(spot_list)

            # scale coordinates for full-res image
            spot_list = spot_list*settings['spot_detection_downsize_factor']
            
            with fluorescent_lock:
                shared_memory_fluorescent[fov_id] = {'spot_indices': spot_list}
            
            output_queue.put(fov_id)
            log_time(fov_id, "Fluorescent Spot Detection", "end")

        except Empty:
            continue

from utils import get_spot_images_from_fov
from model import ResNet, run_model    

def classification_process(segmentation_queue: mp.Queue, fluorescent_queue: mp.Queue, save_queue: mp.Queue, ui_queue: mp.Queue):

    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    segmentation_ready = set()
    fluorescent_ready = set()

    # initalize model
    CHECKPOINT1 = './checkpoint/resnet18_en/version1/best.pt'
    model1 = ResNet('resnet18').to(device=DEVICE)
    model1.load_state_dict(torch.load(CHECKPOINT1))
    model1.eval()

    CHECKPOINT2 = './checkpoint/resnet18_en/version2/best.pt'
    model2 = ResNet('resnet18').to(device=DEVICE)
    model2.load_state_dict(torch.load(CHECKPOINT2))
    model2.eval()

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
     
                segmentation_map = shared_memory_segmentation[fov_id]['segmentation_map']
                spot_list = shared_memory_fluorescent[fov_id]['spot_indices']
                
                dpc_image = shared_memory_dpc[fov_id]['dpc_image']
                fluorescence_image = shared_memory_acquisition[fov_id]['fluorescent'].astype(float)/255
                
                # print(f"Classification Process: getting spot images for FOV {fov_id} with {len(spot_list)} spots")
                cropped_images = get_spot_images_from_fov(fluorescence_image,dpc_image,spot_list,r=15)
                cropped_images = cropped_images.transpose(0, 3, 1, 2)

                scores1 = run_model(model1,DEVICE,cropped_images,4096)[:,1]
                scores2 = run_model(model2,DEVICE,cropped_images,4096)[:,1]

                # use whichever smaller as the final score
                scores = np.minimum(scores1,scores2)

                with classification_lock:
                    shared_memory_classification[fov_id] = {
                        'cropped_images': cropped_images,
                        'scores': scores
                    }
                
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
                    scores = shared_memory_classification[fov_id]['scores']
                    
                    SAVE_NUMPYARRAY = True
                    PATH = 'saved_images'
                    
                    if not SAVE_NUMPYARRAY: 
                        for i, cropped_image in enumerate(cropped_images):
                            filename = os.path.join(PATH, f"{fov_id}_{i}.png")
                            numpy2png(cropped_image,filename)
                            #dpc_image = shared_memory_dpc[fov_id]['dpc_image']*255
                            #save_dpc_image(dpc_image, f'data/{fov_id}.png')
                    else:
                        filename = os.path.join(PATH, f"{fov_id}.npy")
                        np.save(filename, cropped_images)
                        filename = os.path.join(PATH, f"{fov_id}_scores.npy")
                        np.save(filename, scores)

                    temp_dict = shared_memory_final[fov_id]
                    temp_dict['saved'] = True
                    shared_memory_final[fov_id] = temp_dict

                    if shared_memory_final[fov_id]['displayed']:
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
                    temp_dict = shared_memory_final[fov_id]
                    temp_dict['displayed'] = True
                    shared_memory_final[fov_id] = temp_dict

                    if shared_memory_final[fov_id]['saved']:
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
                
                print(f"\n{'=' * 50}")
                print(f"Report for FOV {fov_id}:")
                print(f"RBC counts: {shared_memory_segmentation[fov_id]['n_cells']}, spots: {len(shared_memory_fluorescent[fov_id]['spot_indices'])}")
                print(f"{'=' * 50}")
                report(timing_data, fov_id)

                for shared_memory in [shared_memory_acquisition, shared_memory_dpc, shared_memory_segmentation, shared_memory_fluorescent, shared_memory_classification, shared_memory_final, shared_memory_timing]:
                    if fov_id in shared_memory:
                        del shared_memory[fov_id]
            print(f"FOV {fov_id} memory is freed up")
        except Empty:
            continue



if __name__ == "__main__":
    # Create queues
    dpc_queue = mp.Queue()
    fluorescent_queue = mp.Queue()
    segmentation_queue = mp.Queue()
    fluorescent_detection_queue = mp.Queue()
    classification_queue = mp.Queue()
    save_queue = mp.Queue()
    ui_queue = mp.Queue()
    cleanup_queue = mp.Queue()

    # Create and start processes
    processes = [
        mp.Process(target=image_acquisition, args=(dpc_queue, fluorescent_queue)),
        mp.Process(target=dpc_process, args=(dpc_queue, segmentation_queue)),
        mp.Process(target=fluorescent_spot_detection, args=(fluorescent_queue, fluorescent_detection_queue)),
        mp.Process(target=saving_process, args=(save_queue, cleanup_queue)),
        mp.Process(target=ui_process, args=(ui_queue, cleanup_queue)),
        mp.Process(target=cleanup_process, args=(cleanup_queue,)),
    ]

    for p in processes:
        p.start()


    classification_thread = threading.Thread(target=classification_process, 
                                             args=(classification_queue, fluorescent_detection_queue, save_queue, ui_queue))
    segmentation_thread = threading.Thread(target=segmentation_process,
                                           args=(segmentation_queue, classification_queue)) 

    classification_thread.start()
    segmentation_thread.start()

    try:
        # Wait for all processes to complete
        for p in processes:
            p.join()
        classification_thread.join()
    except KeyboardInterrupt:
        print("Stopping all processes...")
        for p in processes:
            p.terminate()