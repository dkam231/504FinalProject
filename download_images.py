# -*- coding: utf-8 -*-
"""
download_images

Script to retrieve images for the 2024 FathomNet out-of-sample challenge as part of FGVC 10. 

Assumes COCO formated annotation file has been download from http://www.kaggle.com/competitions/fathomnet-out-of-sample-detection
"""
# Author: Eric Orenstein (eorenstein@mbari.org)

import os
import json
import logging
import argparse
from shutil import copyfileobj

import requests
import progressbar


def download_imgs(imgs, outdir=None):
    """
    Download images to an output dir
    
    :param imgs: list of urls 
    :param outdir: desired directory [default to working directory]
    """

    # set the out directory to default if not specified
    if not outdir:
        outdir = os.path.join(os.getcwd(), 'images')

    # make the directory if it does not exist
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        logging.info(f"Created directory {outdir}")

    flag = 0  # keep track of how many image downloaded

    for name, url in progressbar.progressbar(imgs):
        file_name = os.path.join(
            outdir, name
        )

        # only download if the image does not exist in the outdir
        if not os.path.exists(file_name):
            resp = requests.get(url, stream=True)
            resp.raw.decode_content = True
            with open(file_name, 'wb') as f:
                copyfileobj(resp.raw, f)
            flag += 1


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Download images from a COCO annotation file")
    parser.add_argument('dataset', type=str, help='Path to json COCO annotation file')
    parser.add_argument('--outpath', type=str, default=None, help='Path to desired output folder')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    logging.info(f'opening {args.dataset}')
    with open(args.dataset, 'r') as ff:
        dataset = json.load(ff)

    ims = dataset['images']

    logging.info(f'retrieving {len(ims)} images')

    ims = [(im['file_name'], im['coco_url']) for im in ims]

    # download images
    download_imgs(ims, outdir=args.outpath)

    #logging.info(f"Downloaded {flag} new images to {outdir}")