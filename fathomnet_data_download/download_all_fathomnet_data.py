import json
import os
import requests
import cv2
import numpy as np
from fathomnet.api import images as fn_images
from pycocotools import mask as mask_util
from tqdm import tqdm

def process_all_data(json_path, img_dir, mask_dir):
    # 1. Setup Directories
    for d in [img_dir, mask_dir]:
        if not os.path.exists(d): 
            os.makedirs(d)

    # 2. Load JSON Data
    print(f"📂 Loading JSON data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create a map of image_id -> list of annotations
    # (Since one image can have multiple fish/masks)
    ann_map = {}
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann)

    # 3. Process ALL images
    all_images = data.get('images', [])
    print(f"🌊 Starting download and mask generation for ALL {len(all_images)} images...")

    for img_entry in tqdm(all_images):
        img_uuid = img_entry['id']
        file_name = img_entry['file_name']
        img_path = os.path.join(img_dir, file_name)
        mask_output_path = os.path.join(mask_dir, file_name) # Save mask with same name
        
        # --- DOWNLOAD IMAGE ---
        if not os.path.exists(img_path):
            try:
                # Ask API for URL
                dto = fn_images.find_by_uuid(img_uuid)
                r = requests.get(dto.url, timeout=15)
                if r.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(r.content)
            except Exception as e:
                # Skip if download fails (e.g., dead link)
                continue 

        # --- GENERATE PURE TRAINING MASK ---
        # Only generate the mask if it doesn't already exist
        if not os.path.exists(mask_output_path):
            image = cv2.imread(img_path)
            if image is None: 
                continue # Skip if image file is corrupted
            
            # Create an empty black mask the same size as the image
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Find all annotations for this specific image
            if img_uuid in ann_map:
                for ann in ann_map[img_uuid]:
                    rle = ann['segmentation']
                    # Decode the RLE string into pixels (returns 0s and 1s)
                    binary_mask = mask_util.decode(rle)
                    # Combine with other masks in the same image
                    combined_mask = np.maximum(combined_mask, binary_mask)

            # Multiply by 255 so the mask is standard Grayscale (0 = Black/Water, 255 = White/Fish)
            # This is the exact format PyTorch/U-Net expects for Ground Truth labels
            final_mask = combined_mask * 255
            
            # Save the pure mask
            cv2.imwrite(mask_output_path, final_mask)

    print(f"\nDone! \nImages are in: {img_dir}\nTraining Masks are in: {mask_dir}")

if __name__ == "__main__":
    # 1. Get the exact directory where THIS script is saved
    # (/home/jjaemin/504FinalProject/fathomnet_data_download)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go UP one level to the main project folder
    # (/home/jjaemin/504FinalProject)
    project_root = os.path.dirname(script_dir)

    print(project_root)

    # # 3. Process TEST Data using absolute paths
    # print("\n" + "="*40)
    # print("STARTING TEST DATASET")
    # print("="*40)
    # process_all_data(
    #     json_path=os.path.join(project_root, 'fathomnet_data_download', 'test.json'), 
    #     img_dir=os.path.join(project_root, 'images_seg', 'test'), 
    #     mask_dir=os.path.join(project_root, 'masks_seg', 'test')
    # )

    # 4. Process TRAIN Data using absolute paths
    print("\n" + "="*40)
    print("STARTING TRAIN DATASET")
    print("="*40)
    process_all_data(
        json_path=os.path.join(project_root, 'fathomnet_data_download', 'train.json'), 
        img_dir=os.path.join(project_root, 'images_seg', 'train'), 
        mask_dir=os.path.join(project_root, 'masks_seg', 'train')
    )