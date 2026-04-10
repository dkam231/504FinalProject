import json
import os
import requests
import cv2
import numpy as np
from fathomnet.api import images as fn_images
from pycocotools import mask as mask_util
from tqdm import tqdm

def process_all_data(json_path, img_dir, mask_dir):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"\n📂 Loading JSON data from: {json_path}")
    print(f"📁 Image output dir: {img_dir}")
    print(f"📁 Mask output dir : {mask_dir}")
    print(f"✅ JSON exists? {os.path.exists(json_path)}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    all_images = data.get("images", [])

    print(f"📝 Number of annotations: {len(annotations)}")
    print(f"🖼️  Number of images in JSON: {len(all_images)}")

    ann_map = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann)

    download_success = 0
    download_failed = 0
    already_have_image = 0
    mask_written = 0
    mask_already_exists = 0
    image_read_failed = 0
    no_annotations_for_image = 0

    print(f"🌊 Starting download and mask generation for ALL {len(all_images)} images...")

    for img_entry in tqdm(all_images):
        img_id = img_entry["id"]
        file_name = img_entry["file_name"]

        if isinstance(img_id, str) and "-" in img_id:
            img_uuid = img_id
        else:
            img_uuid = os.path.splitext(file_name)[0]

        img_path = os.path.join(img_dir, file_name)
        mask_output_path = os.path.join(mask_dir, file_name)

        if not os.path.exists(img_path):
            try:
                dto = fn_images.find_by_uuid(img_uuid)
                if dto is None or not hasattr(dto, "url") or dto.url is None:
                    print(f"[DOWNLOAD FAIL] No URL returned for file={file_name} uuid={img_uuid}")
                    continue

                r = requests.get(dto.url, timeout=30)
                if r.status_code == 200:
                    with open(img_path, "wb") as f:
                        f.write(r.content)
                else:
                    print(f"[DOWNLOAD FAIL] status={r.status_code} file={file_name} url={dto.url}")
                    continue

            except Exception as e:
                print(f"[DOWNLOAD EXCEPTION] file={file_name} uuid={img_uuid} error={repr(e)}")
                continue
        else:
            already_have_image += 1

        if not os.path.exists(img_path):
            print(f"[MISSING IMAGE AFTER DOWNLOAD] {img_path}")
            download_failed += 1
            continue

        if os.path.exists(mask_output_path):
            mask_already_exists += 1
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"[IMAGE READ FAIL] Could not read image: {img_path}")
            image_read_failed += 1
            continue

        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if img_uuid in ann_map:
            for ann in ann_map[img_uuid]:
                try:
                    rle = ann["segmentation"]
                    binary_mask = mask_util.decode(rle)

                    if len(binary_mask.shape) == 3:
                        binary_mask = binary_mask[:, :, 0]

                    binary_mask = (binary_mask > 0).astype(np.uint8)

                    combined_mask = np.maximum(combined_mask, binary_mask)

                except Exception as e:
                    print(f"[MASK DECODE FAIL] file={file_name} uuid={img_uuid} error={repr(e)}")
        else:
            no_annotations_for_image += 1

        final_mask = combined_mask * 255

        ok = cv2.imwrite(mask_output_path, final_mask)
        if ok:
            mask_written += 1
        else:
            print(f"[MASK WRITE FAIL] {mask_output_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Images dir              : {img_dir}")
    print(f"Masks dir               : {mask_dir}")
    print(f"Downloaded successfully : {download_success}")
    print(f"Download failed         : {download_failed}")
    print(f"Already had image       : {already_have_image}")
    print(f"Masks written           : {mask_written}")
    print(f"Masks already existed   : {mask_already_exists}")
    print(f"Image read failed       : {image_read_failed}")
    print(f"No annotations for img  : {no_annotations_for_image}")

    num_img_files = len([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
    num_mask_files = len([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])

    print(f"Actual files in img_dir : {num_img_files}")
    print(f"Actual files in mask_dir: {num_mask_files}")
    print("=" * 60)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print(f"Project root: {project_root}")

<<<<<<< HEAD
    print("\n" + "=" * 40)
    print("STARTING TEST DATASET")
    print("=" * 40)
    process_all_data(
        json_path=os.path.join(project_root, "fathomnet_segmentations", "test.json"),
        img_dir=os.path.join(project_root, "images_seg", "test"),
        mask_dir=os.path.join(project_root, "masks_seg", "test"),
    )
=======
    # # 3. Process TEST Data using absolute paths
    # print("\n" + "="*40)
    # print("STARTING TEST DATASET")
    # print("="*40)
    # process_all_data(
    #     json_path=os.path.join(project_root, 'fathomnet_data_download', 'test.json'), 
    #     img_dir=os.path.join(project_root, 'images_seg', 'test'), 
    #     mask_dir=os.path.join(project_root, 'masks_seg', 'test')
    # )
>>>>>>> 666f59335bc316546a2fcaa31064616111a2bb8c


    print("\n" + "=" * 40)
    print("STARTING TRAIN DATASET")
    print("=" * 40)
    process_all_data(
<<<<<<< HEAD
        json_path=os.path.join(project_root, "fathomnet_segmentations", "train.json"),
        img_dir=os.path.join(project_root, "images_seg", "train"),
        mask_dir=os.path.join(project_root, "masks_seg", "train"),
=======
        json_path=os.path.join(project_root, 'fathomnet_data_download', 'train.json'), 
        img_dir=os.path.join(project_root, 'images_seg', 'train'), 
        mask_dir=os.path.join(project_root, 'masks_seg', 'train')
>>>>>>> 666f59335bc316546a2fcaa31064616111a2bb8c
    )