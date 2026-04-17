import json
import os
import requests
from fathomnet.api import images as fn_images
from tqdm import tqdm

def download_images(json_path, out_dir):
    # 1. Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"📁 Created directory: {out_dir}")

    # 2. Load the JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_entries = data.get('images', [])
    print(f"🔍 Found {len(image_entries)} entries. Starting verified download...")

    # 3. Download Loop
    for img in tqdm(image_entries):
        img_uuid = img['id']
        file_name = img['file_name']
        save_path = os.path.join(out_dir, file_name)

        if os.path.exists(save_path):
            continue

        # Try API lookup first (most reliable)
        try:
            # This gets the current verified URL from FathomNet
            dto = fn_images.find_by_uuid(img_uuid)
            url = dto.url
            
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(r.content)
                continue # Success!
        except:
            pass # Fallback to manual if API fails

        # Fallback: Many challenge images are stored in this CDN path
        fallback_url = f"https://database.fathomnet.org/static/m3/media/{img_uuid}.png"
        try:
            r = requests.get(fallback_url, timeout=5)
            if r.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(r.content)
        except:
            print(f"⚠️ Could not find image: {img_uuid}")

if __name__ == "__main__":
    # Point this to your test.json
    download_images('fathomnet_segmentations/test.json', 'images_seg')