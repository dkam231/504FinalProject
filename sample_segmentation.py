import json
import os
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fathomnet.api import images as fn_images
from pycocotools import mask as mask_util
from tqdm import tqdm

def process_sample(json_path, img_dir, mask_dir, count=100):
    # 1. Setup Directories
    for d in [img_dir, mask_dir]:
        if not os.path.exists(d): os.makedirs(d)

    # 2. Load JSON Data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create a map of image_id -> list of annotations
    # (Since one image can have multiple fish/masks)
    ann_map = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann)

    # 3. Process the first 'count' images
    sample_images = data['images'][:count]
    print(f"🌊 Starting download and mask generation for {count} images...")

    for img_entry in tqdm(sample_images):
        img_uuid = img_entry['id']
        file_name = img_entry['file_name']
        img_path = os.path.join(img_dir, file_name)
        
        # --- DOWNLOAD IMAGE ---
        if not os.path.exists(img_path):
            try:
                # Ask API for URL
                dto = fn_images.find_by_uuid(img_uuid)
                r = requests.get(dto.url, timeout=10)
                if r.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(r.content)
            except:
                continue # Skip if download fails

        # --- GENERATE MASK ---
        image = cv2.imread(img_path)
        if image is None: continue
        
        # Create an empty black mask the same size as the image
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Find all annotations for this specific image
        if img_uuid in ann_map:
            for ann in ann_map[img_uuid]:
                rle = ann['segmentation']
                # Decode the RLE string into pixels
                binary_mask = mask_util.decode(rle)
                # Combine with other masks in the same image
                combined_mask = np.maximum(combined_mask, binary_mask)

        # --- CREATE VISUAL OVERLAY ---
        # We'll make the mask red
        overlay = image.copy()
        overlay[combined_mask > 0] = [0, 0, 255] # BGR for Red
        
        # Blend the original image and the red mask (50% transparency)
        visual_check = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Save the result
        mask_output_path = os.path.join(mask_dir, f"mask_{file_name}")
        cv2.imwrite(mask_output_path, visual_check)

    print(f"\n✅ Done! \nImages are in: {img_dir}\nOverlaid masks are in: {mask_dir}")

if __name__ == "__main__":
    process_sample('fathomnet_segmentations/test.json', 'sample_100', 'sample_masks', count=100)