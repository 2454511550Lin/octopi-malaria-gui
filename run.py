#import multiprocessing as mp
#from multiprocessing import Lock

import torch.multiprocessing as mp
import numpy as np
from queue import Empty
import time

from simulation import get_image
import threading
from log import report

from utils import SharedConfig, numpy2png

import os

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
shared_memory_timing['START'] = None
shared_memory_timing['END'] = None

# Existing locks
dpc_lock = manager.Lock()
segmentation_lock = manager.Lock()
fluorescent_lock = manager.Lock()
classification_lock = manager.Lock()
final_lock = manager.Lock()

# time lock
timing_lock = manager.Lock()
timeout = 0.1
shared_config = SharedConfig()
shared_config.set_path('data')

INIT_FOCUS_RANGE_START_MM = 6.2
INIT_FOCUS_RANGE_END_MM = 6.7
SCAN_FOCUS_SEARCH_RANGE_MM = 0.1

import cv2

def log_time(fov_id: str, process_name: str, event: str):
    with timing_lock:
        #main_logger.info(f"Logging time for FOV {fov_id} in {process_name} at {event}")

        if shared_memory_timing['START'] is None:
            shared_memory_timing['START'] = time.time()
        shared_memory_timing['END'] = time.time()

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
from control.utils import generate_scan_grid,interpolate_focus

def image_acquisition_simulation(dpc_queue: mp.Queue, fluorescent_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):

    image_iterator = get_image()

    counter = 0 

    BEGIN = time.time()
    while not shutdown_event.is_set():

        if not start_event.is_set():
            time.sleep(1)
            continue     
        
        logger = shared_config.setup_process_logger()

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

            left_half = crop_image(left_half)
            right_half = crop_image(right_half)
            fluorescent = crop_image(fluorescent)
    
            shared_memory_acquisition[fov_id] = {
                'left_half': left_half,
                'right_half': right_half,
                'fluorescent': fluorescent
            }

            with final_lock:
                if shared_config.save_raw_images.value:
                    save_path = shared_config.get_path()
                    if shared_config.SAVE_NPY.value:
                        left_filename = f"{fov_id}_left_half.npy"
                        right_filename = f"{fov_id}_right_half.npy"
                        fluorescent_filename = f"{fov_id}_fluorescent.npy"
                        np.save(os.path.join(save_path, left_filename), left_half)
                        np.save(os.path.join(save_path, right_filename), right_half)
                        np.save(os.path.join(save_path, fluorescent_filename), fluorescent)
                    else:
                    # save the bmp
                        left_filename = f"{fov_id}_left_half.bmp"
                        right_filename = f"{fov_id}_right_half.bmp"
                        fluorescent_filename = f"{fov_id}_fluorescent.bmp"
                        cv2.imwrite(os.path.join(save_path, left_filename), left_half)
                        cv2.imwrite(os.path.join(save_path, right_filename), right_half)
                        cv2.imwrite(os.path.join(save_path, fluorescent_filename), fluorescent)

            dpc_queue.put(fov_id)
            fluorescent_queue.put(fov_id)

            log_time(fov_id, "Image Acquisition", "end")
        
        except StopIteration:
            logger.info("No more images to process")
            break
        
        #logger.info(f"Image Acquisition: Processed FOV {fov_id}")
        time.sleep(1) 

        counter += 1
        if counter == 50:
            counter = 0
            while start_event.is_set() and not shutdown_event.is_set():
                time.sleep(1)
            
from microscope import Microscope
def image_acquisition(dpc_queue: mp.Queue, fluorescent_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):

    simulation = False
    microscope = Microscope(is_simulation=simulation)

    microscope.camera.start_streaming()
    microscope.camera.set_software_triggered_acquisition()
    microscope.camera.disable_callback()

    microscope.home_xyz()
    
    live_channel_index = -1

    while not shutdown_event.is_set():

        # check if to loading or to scanning position
        if not start_event.is_set():
            with shared_config.position_lock:
                if shared_config.to_scanning.value and not shared_config.to_loading.value:
                    #main_logger.info("Moving to scanning position")
                    microscope.to_scanning_position()
                    shared_config.reset_to_scanning()
                    continue
                if shared_config.to_loading.value and not shared_config.to_scanning.value:
                    #main_logger.info("Moving to loading position")
                    microscope.to_loading_position()
                    shared_config.reset_to_loading()
                    continue

            #main_logger.info("Check the live view active vlaue: ", shared_config.is_live_view_active.value)
            
            if shared_config.is_live_view_active.value:

                if live_channel_index != shared_config.live_channel_selected.value:
                    live_channel_index = shared_config.live_channel_selected.value
                    # get the value from manager.list
                    microscope.set_channel(shared_config.live_channels_list[live_channel_index])

                image = microscope.acquire_image()

                #main_logger.info("Live view image shape: ", image.shape)
                
                shared_config.set_live_view_image(crop_image(image))
            
            time.sleep(1/shared_config.frame_rate.value)
            
            continue
        
        logger = shared_config.setup_process_logger()

        try:
            # do auto focus
            # set the chanel to BF
            microscope.set_channel("BF LED matrix left half")
            microscope.run_autofocus(step_size_mm = [0.01, 0.001, 0.0015], start_z_mm = INIT_FOCUS_RANGE_START_MM, end_z_mm = INIT_FOCUS_RANGE_END_MM)

            # generate the focus map
            # scan settings
            dx_mm = 0.9
            dy_mm = 0.9
            Nx = shared_config.nx.value
            Ny = shared_config.ny.value
            Nx_focus = 2
            Ny_focus = 2
            offset_x_mm = microscope.get_x()
            offset_y_mm = microscope.get_y()
            offset_z_mm = microscope.get_z()

            # generate scan grid
            scan_grid = generate_scan_grid(dx_mm, dy_mm, Nx, Ny, offset_x_mm, offset_y_mm, S_scan=True)

            # generate focus map
            x = offset_x_mm + np.linspace(0, (Nx - 1) * dx_mm, Nx_focus)
            y = offset_y_mm + np.linspace(0, (Ny - 1) * dy_mm, Ny_focus)
            focus_map = []
            microscope.set_channel("BF LED matrix left half")
            for yi in y:
                microscope.move_y_to(yi)
                for xi in x:
                    microscope.move_x_to(xi)
                    z_focus = microscope.run_autofocus(step_size_mm = [0.01, 0.0015], start_z_mm = offset_z_mm - SCAN_FOCUS_SEARCH_RANGE_MM/2, end_z_mm = offset_z_mm + SCAN_FOCUS_SEARCH_RANGE_MM/2)
                    focus_map.append((xi, yi, z_focus))
                    offset_z_mm = z_focus
            #logger.info("Focus map: ",focus_map)
            z_map = interpolate_focus(scan_grid, focus_map)

            # scan using focus map
            prev_x, prev_y = None, None
            coordinates = []
            for i, ((x, y), z) in enumerate(zip(scan_grid, z_map)):
                if x != prev_x:
                    microscope.move_x_to(x)
                    prev_x = x
                if y != prev_y:
                    microscope.move_y_to(y)
                    prev_y = y
                microscope.move_z_to(z)
                fov_id = f"{i}"
                log_time(fov_id, "Image Acquisition", "start")
                channels = ["BF LED matrix left half","BF LED matrix right half","Fluorescence 405 nm Ex"]

                microscope.set_channel(channels[0])
                left_half = microscope.acquire_image()
                microscope.set_channel(channels[1])
                right_half = microscope.acquire_image()
                microscope.set_channel(channels[2])
                fluorescent = microscope.acquire_image()

                shared_config.set_live_view_image(crop_image(left_half))
                
                if not simulation:
                    # go from 3k x 3k x 3 to 3k x 3k, take the middle channel
                    left_half = left_half[:,:,1]
                    right_half = right_half[:,:,1]
                    # for fluorescent (3k x 3k x 3), we reverse the channel order
                    fluorescent = fluorescent[:, :, ::-1]
                else:
                    fluorescent = np.stack([fluorescent,fluorescent,fluorescent],axis=2)

                left_half = crop_image(left_half)
                right_half = crop_image(right_half)
                fluorescent = crop_image(fluorescent)

                shared_memory_acquisition[fov_id] = {
                    'left_half': left_half,
                    'right_half': right_half,
                    'fluorescent': fluorescent
                }

                with final_lock:
                    if shared_config.save_raw_images.value:
                        save_path = shared_config.get_path()
                        if shared_config.SAVE_NPY.value:
                            left_filename = f"{fov_id}_left_half.npy"
                            right_filename = f"{fov_id}_right_half.npy"
                            fluorescent_filename = f"{fov_id}_fluorescent.npy"
                            np.save(os.path.join(save_path, left_filename), left_half)
                            np.save(os.path.join(save_path, right_filename), right_half)
                            np.save(os.path.join(save_path, fluorescent_filename), fluorescent)
                        else:
                        # save the bmp
                            left_filename = f"{fov_id}_left_half.bmp"
                            right_filename = f"{fov_id}_right_half.bmp"
                            fluorescent_filename = f"{fov_id}_fluorescent.bmp"
                            cv2.imwrite(os.path.join(save_path, left_filename), left_half)
                            cv2.imwrite(os.path.join(save_path, right_filename), right_half)
                            cv2.imwrite(os.path.join(save_path, fluorescent_filename), fluorescent)
    
                dpc_queue.put(fov_id)
                fluorescent_queue.put(fov_id)

                log_time(fov_id, "Image Acquisition", "end")
        except Exception as e:
            logger.error(f"Error in image acquisition: {e}")
            continue

        while start_event.is_set() and not shutdown_event.is_set():
            time.sleep(1)

    microscope.close()
    print("Image acquisition process finished")

from utils import generate_dpc,save_dpc_image

def dpc_process(input_queue: mp.Queue, output_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    while not shutdown_event.is_set():

        start_event.wait()
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "DPC Process", "start")
            
            data = shared_memory_acquisition[fov_id]
            left_half = data['left_half'].astype(np.float16)/255
            right_half = data['right_half'].astype(np.float16)/255
            
            dpc_image = generate_dpc(left_half, right_half,use_gpu=False) 

            assert dpc_image.shape == (2800, 2800)
            
            with dpc_lock:
                shared_memory_dpc[fov_id] = {'dpc_image': dpc_image}        
            
            output_queue.put(fov_id)
            log_time(fov_id, "DPC Process", "end")
        except Empty:
            continue
        except Exception as e:
            logger = shared_config.setup_process_logger()
            logger.error(f"Unknown error in DPC process {e}")
            continue

    print("DPC process finished")

def segmentation_process(input_queue: mp.Queue, output_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    from interactive_m2unet_inference import M2UnetInteractiveModel as m2u
    import torch
    from scipy.ndimage import label

    model_path = 'checkpoint/m2unet_model_flat_erode1_wdecay5_smallbatch/model_4000_11.pth'
    model = m2u(pretrained_model=model_path, use_trt=False)
    
    while not shutdown_event.is_set():
        start_event.wait()
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "Segmentation Process", "start")
            
            # Wait for DPC to finish
            while fov_id not in shared_memory_dpc:
                time.sleep(timeout)
            
            dpc_image = shared_memory_dpc[fov_id]['dpc_image']
            # convert dpc to np.int8
            dpc_image = (dpc_image*255).astype(np.uint8)
            
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
        except Exception as e:
            logger = shared_config.setup_process_logger()
            logger.error(f"Unknown error in segmentation process {e}")
            with segmentation_lock:
                shared_memory_segmentation[fov_id] = {'segmentation_map': np.zeros((2800, 2800)), 'n_cells': 0}
            continue

from utils import remove_background, resize_image_cp, detect_spots, prune_blobs, settings, seg_spot_filter_one_fov
MAX_SPOTS_THRESHOLD = 5000  # Maximum number of spots allowed

def fluorescent_spot_detection(input_queue: mp.Queue, output_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    
    while not shutdown_event.is_set():
        start_event.wait()
        logger = shared_config.setup_process_logger()
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "Fluorescent Spot Detection", "start")
            
            fluorescent = shared_memory_acquisition[fov_id]['fluorescent']

            I_fluorescence_bg_removed = remove_background(fluorescent,return_gpu_image=False)

            spot_list = detect_spots(resize_image_cp(I_fluorescence_bg_removed,
                                                     downsize_factor=settings['spot_detection_downsize_factor']),
                                                     thresh=settings['spot_detection_threshold'])      
            
            if len(spot_list) > MAX_SPOTS_THRESHOLD:
                logger.info(f"Abnormal number of fluorescent spots detected in FOV {fov_id}: {len(spot_list)}")
                spot_list = spot_list[:MAX_SPOTS_THRESHOLD]
            
            if len(spot_list) > 0:
                spot_list = prune_blobs(spot_list)

            spot_list = spot_list*settings['spot_detection_downsize_factor']
            
            with fluorescent_lock:
                    shared_memory_fluorescent[fov_id] = {
                        'spot_indices': spot_list,
                        'abnormal_spots': len(spot_list) > MAX_SPOTS_THRESHOLD,
                        'spot_count': len(spot_list)
                    }

            # free the memory
            del I_fluorescence_bg_removed
            
            output_queue.put(fov_id)
            log_time(fov_id, "Fluorescent Spot Detection", "end")

        except Empty:
            continue
        except Exception as e:
            logger.error(f"Unknown error in fluorescent spot detection {e}")
            with fluorescent_lock:
                shared_memory_fluorescent[fov_id] = {
                    'spot_indices': [],
                    'abnormal_spots': False,
                    'spot_count': 0
                }
            continue
    
    print("Fluorescent spot detection process finished")

from utils import get_spot_images_from_fov
from model import ResNet, run_model    

def classification_process(segmentation_queue: mp.Queue, fluorescent_queue: mp.Queue, save_queue: mp.Queue, ui_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):

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

    while not shutdown_event.is_set():
        start_event.wait()
        logger = shared_config.setup_process_logger()
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

                if len(spot_list) > 0:

                    filtered_spots = seg_spot_filter_one_fov(segmentation_map, spot_list)
                
                    dpc_image = shared_memory_dpc[fov_id]['dpc_image']
                    fluorescence_image = shared_memory_acquisition[fov_id]['fluorescent'].astype(np.float16)/255
                
                    cropped_images = get_spot_images_from_fov(fluorescence_image,dpc_image,filtered_spots,r=15)
                    cropped_images = cropped_images.transpose(0, 3, 1, 2)

                    scores1 = run_model(model1,DEVICE,cropped_images,1024)[:,1]
                    scores2 = run_model(model2,DEVICE,cropped_images,1024)[:,1]

                    # use whichever smaller as the final score
                    scores = np.minimum(scores1,scores2)
                else:
                    filtered_spots = np.array([])
                    scores = np.array([])
                    cropped_images = np.array([])


                with classification_lock:
                    shared_memory_classification[fov_id] = {
                        'cropped_images': cropped_images,
                        'scores': scores,
                        'filtered_spots_count': len(filtered_spots)
                    }

                #del cropped_images, scores
                
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

        except Empty:
            continue
        except Exception as e:
            logger.error(f"Unknown error in classification process {e}")
            with classification_lock:
                shared_memory_classification[fov_id] = {
                    'cropped_images': [],
                    'scores': [],
                    'filtered_spots_count': 0
                }
            continue
    print("Classification process finished")


from utils import numpy2png
import os

def saving_process(input_queue: mp.Queue, output: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    
    while not shutdown_event.is_set():

        start_event.wait()
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "Saving Process", "start")
            
            with final_lock:
                if fov_id in shared_memory_final and not shared_memory_final[fov_id]['saved']:
                    
                    # save the cropped images to png
                    cropped_images = (shared_memory_classification[fov_id]['cropped_images']*255).astype(np.uint8)
                    scores = shared_memory_classification[fov_id]['scores']
                    
                    save_path = shared_config.get_path()
                    
                    if shared_config.save_spot_images.value:
                        filename = os.path.join(save_path, f"{fov_id}.npy")
                        np.save(filename, cropped_images)
                        filename = os.path.join(save_path, f"{fov_id}_scores.npy")
                        np.save(filename, scores)
                    if shared_config.save_overlay_images.value:
                        #filename = os.path.join(save_path, f"{fov_id}_overlay.npy")
                        #fluorescent_image = shared_memory_acquisition[fov_id]['fluorescent']
                        filename = os.path.join(save_path, f"{fov_id}_dpc.npy")
                        dpc_image = shared_memory_dpc[fov_id]['dpc_image']
                        #fluorescent_image_int8 = fluorescent_image.astype(np.uint8)
                        dpc_image_int8 = (dpc_image*255).astype(np.uint8)
                        #img = np.stack([fluorescent_image_int8[:,:,0], fluorescent_image_int8[:,:,1], fluorescent_image_int8[:,:,2], dpc_image_int8], axis=0)
                        if shared_config.SAVE_NPY.value:
                            np.save(filename, dpc_image_int8)
                        else:
                            filename = os.path.join(save_path, f"{fov_id}_dpc.bmp")
                            cv2.imwrite(filename, dpc_image_int8)

                    temp_dict = shared_memory_final[fov_id]
                    temp_dict['saved'] = True
                    shared_memory_final[fov_id] = temp_dict

                    if shared_memory_final[fov_id]['displayed']:
                        output.put(fov_id)
            
                    log_time(fov_id, "Saving Process", "end")
        
        except Empty:
            continue
        except Exception as e:
            logger = shared_config.setup_process_logger()
            logger.error(f"Unknown error in saving process {e}")
            continue

    print("Saving process finished")
        

def cleanup_process(cleanup_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    while not shutdown_event.is_set():
        start_event.wait()
        
        try:
            fov_id = cleanup_queue.get(timeout=timeout)
            
            # Clean up shared memory
            with final_lock, timing_lock:

                # Calculate processing times and generate visualization
                timing_data = shared_memory_timing.get(fov_id, {})

                logger = shared_config.setup_process_logger()
                logger.info(f"\n{'=' * 50}")
                logger.info(f"Report for FOV {fov_id}:")
                logger.info(f"RBC counts: {shared_memory_segmentation[fov_id]['n_cells']}, spots: {len(shared_memory_fluorescent[fov_id]['spot_indices'])},filtered spots: {shared_memory_classification[fov_id]['filtered_spots_count']}")
                logger.info(f"{'=' * 50}")
                report(timing_data, fov_id, logger)

                shared_memories = [shared_memory_acquisition, shared_memory_dpc, shared_memory_segmentation, 
                                   shared_memory_fluorescent, shared_memory_classification, shared_memory_final, 
                                   shared_memory_timing]
                
                for shared_memory in shared_memories:
                    if fov_id in shared_memory:
                        del shared_memory[fov_id]
                
                # Confirm deletion
                if all(fov_id not in shared_memory for shared_memory in shared_memories):
                    logger.info(f"FOV {fov_id} memory has been successfully freed up")
                else:
                    logger.error(f"Warning: FOV {fov_id} may not have been completely removed from all shared memories")
        except Empty:
            continue
        except Exception as e:
            logger = shared_config.setup_process_logger()
            logger.error(f"Unknown error in cleanup process {e}")
            continue

    print("Cleanup process finished")

from ui import ui_process

if __name__ == "__main__":

    # get the first cmd argument
    import sys
    if len(sys.argv) > 1:
        simulation = sys.argv[1] == "simulation"
    else:
        simulation = False

    # Create queues
    dpc_queue = mp.Queue()
    fluorescent_queue = mp.Queue()
    segmentation_queue = mp.Queue()
    fluorescent_detection_queue = mp.Queue()
    classification_queue = mp.Queue()
    save_queue = mp.Queue()
    ui_queue = mp.Queue()
    cleanup_queue = mp.Queue()

    # Create an event to signal shutdown
    shutdown_event = mp.Event()
    # Create an event to signal start
    start_event = mp.Event()

    # Create and start processes
    processes = [
        mp.Process(target=image_acquisition_simulation if simulation else image_acquisition, args=(dpc_queue, fluorescent_queue, shutdown_event,start_event), name="Image Acquisition"),
        mp.Process(target=dpc_process, args=(dpc_queue, segmentation_queue, shutdown_event,start_event), name="DPC Process"),
        mp.Process(target=fluorescent_spot_detection, args=(fluorescent_queue, fluorescent_detection_queue, shutdown_event,start_event), name="Fluorescent Spot Detection"),
        mp.Process(target=saving_process, args=(save_queue, cleanup_queue, shutdown_event,start_event), name="Saving Process"),
        mp.Process(target=cleanup_process, args=(cleanup_queue, shutdown_event,start_event), name="Cleanup Process")
    ]

    # Start the UI
    ui_process = mp.Process(target=ui_process, args=(ui_queue, cleanup_queue, shared_memory_final, shared_memory_classification, 
                                            shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, 
                                            shared_memory_timing, final_lock, timing_lock,start_event,shutdown_event,shared_config), name="UI Process")

    ui_process.start()

    for p in processes:
        p.start()

    classification_thread = threading.Thread(target=classification_process, 
                                             args=(classification_queue, fluorescent_detection_queue, save_queue, ui_queue, shutdown_event,start_event),name="Classification Process")
    segmentation_thread = threading.Thread(target=segmentation_process,
                                           args=(segmentation_queue, classification_queue, shutdown_event,start_event),name="Segmentation Process")

    classification_thread.start()
    segmentation_thread.start()

    #logger = shared_config.setup_process_logger()

    try:
        while not shutdown_event.is_set():
            time.sleep(1)    
    except KeyboardInterrupt:
        #logger.info("Stopping all processes...")
        print("Stopping all processes...")
    finally:
        shutdown_event.set()
        for p in processes:
            if p.name == "Image Acquisition":
                p.join(timeout=2)  # Give image acquisition process more time to shut down
            else:
                p.join(timeout=1)  # Give other processes more time to shut down
            if p.is_alive():
                #logger.info(f"Force terminating process {p.name}")
                print(f"Force terminating process {p.name}")
                p.terminate()
        classification_thread.join(timeout=1)
        segmentation_thread.join(timeout=1)
        ui_process.join(timeout=1)
        if ui_process.is_alive():
            #logger.info("Force terminating UI process")
            print("Force terminating UI process")
            ui_process.terminate()
        #logger.info("All processes have been shut down.")
        print("All processes have been shut down.")
        if classification_thread.is_alive() or segmentation_thread.is_alive():
            #logger.info("Force terminating classification and segmentation threads")
            print("Force terminating classification and segmentation threads")
            import os
            os._exit(0)