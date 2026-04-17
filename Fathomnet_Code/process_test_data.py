import json
import os
import requests
from tqdm import tqdm


def start_download(json_path, out_dir):
    # 1. Create the folder if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 2. Open your test.json
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    print(f"🔍 Found {len(images)} image entries in JSON.")

    # 3. Loop and download
    for img in tqdm(images, desc="Downloading images"):
        img_id = img['id']
        file_name = img['file_name']
        save_path = os.path.join(out_dir, file_name)

        if os.path.exists(save_path):
            continue # Skip if already there

        # Construct the URL manually since it's missing from your JSON
        url = f"https://fathomnet.org/static/m3/media/{img_id}.png"
        
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                # Try .jpg if .png doesn't exist
                r = requests.get(url.replace('.png', '.jpg'), timeout=10)
            
            if r.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(r.content)
        except:
            pass

if __name__ == "__main__":
    # Point this to your test.json file
    start_download('fathomnet_segmentations/test.json', 'images_seg')