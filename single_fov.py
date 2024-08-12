import numpy as np
import cv2
from scipy.ndimage import label
import torch

from utils import generate_dpc, remove_background, resize_image_cp, detect_spots, prune_blobs, settings, seg_spot_filter_one_fov, get_spot_images_from_fov
from interactive_m2unet_inference import M2UnetInteractiveModel as m2u
from model import ResNet, run_model

def process_single_fov(fov_id, base_path):
    # Load the three separate BMP files
    left_half = cv2.imread(f"{base_path}/{fov_id}_BF_LED_matrix_left_half.bmp", cv2.IMREAD_UNCHANGED)
    right_half = cv2.imread(f"{base_path}/{fov_id}_BF_LED_matrix_right_half.bmp", cv2.IMREAD_UNCHANGED)
    fluorescent = cv2.imread(f"{base_path}/{fov_id}_Fluorescence_405_nm_Ex.bmp", cv2.IMREAD_UNCHANGED)

    # Ensure images are 2D (grayscale)
    if len(left_half.shape) == 3:
        left_half = left_half[:,:,0]
    if len(right_half.shape) == 3:
        right_half = right_half[:,:,0]
    
    # For fluorescent, we want to keep it as a 3-channel image
    if len(fluorescent.shape) == 2:
        fluorescent = np.stack([fluorescent, fluorescent, fluorescent], axis=2)

    # DPC Process
    left_half = left_half.astype(np.float16) / 255
    right_half = right_half.astype(np.float16) / 255
    dpc_image = generate_dpc(left_half, right_half, use_gpu=False)

    # Segmentation Process
    model_path = 'checkpoint/m2unet_model_flat_erode1_wdecay5_smallbatch/model_4000_11.pth'
    segmentation_model = m2u(pretrained_model=model_path, use_trt=False)
    dpc_uint8 = (dpc_image * 255).astype(np.uint8)
    result = segmentation_model.predict_on_images(dpc_uint8)
    threshold = 0.5
    segmentation_mask = (255 * (result > threshold)).astype(np.uint8)
    _, n_cells = label(segmentation_mask)

    # Fluorescent Spot Detection
    I_fluorescence_bg_removed = remove_background(fluorescent, return_gpu_image=False)
    spot_list = detect_spots(resize_image_cp(I_fluorescence_bg_removed,
                                             downsize_factor=settings['spot_detection_downsize_factor']),
                             thresh=settings['spot_detection_threshold'])

    if len(spot_list) > 0:
        spot_list = prune_blobs(spot_list)

    spot_list = spot_list * settings['spot_detection_downsize_factor']

    # Classification Process
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    CHECKPOINT1 = './checkpoint/resnet18_en/version1/best.pt'
    model1 = ResNet('resnet18').to(device=DEVICE)
    model1.load_state_dict(torch.load(CHECKPOINT1))
    model1.eval()

    CHECKPOINT2 = './checkpoint/resnet18_en/version2/best.pt'
    model2 = ResNet('resnet18').to(device=DEVICE)
    model2.load_state_dict(torch.load(CHECKPOINT2))
    model2.eval()

    filtered_spots = seg_spot_filter_one_fov(segmentation_mask, spot_list)
    fluorescence_image = fluorescent.astype(np.float16) / 255

    if len(filtered_spots) > 0:
        cropped_images = get_spot_images_from_fov(fluorescence_image, dpc_image, filtered_spots, r=15)
        cropped_images = cropped_images.transpose(0, 3, 1, 2)

        scores1 = run_model(model1, DEVICE, cropped_images, 1024)[:, 1]
        scores2 = run_model(model2, DEVICE, cropped_images, 1024)[:, 1]

        scores = np.minimum(scores1, scores2)
    else:
        cropped_images = np.array([])
        scores = np.array([])

    # Return results
    return {
        'dpc_image': dpc_image,
        'segmentation_mask': segmentation_mask,
        'n_cells': n_cells,
        'spot_list': spot_list,
        'filtered_spots': filtered_spots,
        'cropped_images': cropped_images,
        'scores': scores
    }

if __name__ == "__main__":
    base_path = "data/pat"  # Replace with the actual path to your image folder
    fov_id = "0_7_0"  # Replace with the actual FOV ID
    results = process_single_fov(fov_id, base_path)
    
    print(f"Number of cells detected: {results['n_cells']}")
    print(f"Number of spots detected: {len(results['spot_list'])}")
    print(f"Number of filtered spots: {len(results['filtered_spots'])}")
    print(f"Number of classified spots: {len(results['scores'])}")
    print(f"Number of cropped images: {len(results['cropped_images'])}")