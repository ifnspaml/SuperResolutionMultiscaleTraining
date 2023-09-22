import argparse
import glob
import logging
import os.path
from collections import defaultdict

import cv2
import numpy as np
import pandas
import torch

from PIL import Image, ImageOps

from pathlib import Path
from tqdm import tqdm
from models.select_model import define_Model
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from torch.utils.data import DataLoader

from mikrosr.validation.matching import get_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#print("starting debug")
#import pydevd_pycharm
#pydevd_pycharm.settrace('134.169.30.95', port=56789, stdoutToServer=True, stderrToServer=True)
#print("after debug")


def main():
    parser = argparse.ArgumentParser("Evaluate the bicubic baseline on a testset")
    parser.add_argument("--lr")
    parser.add_argument("--hr")
    parser.add_argument("--out")
    parser.add_argument("--save-img")
    parser.add_argument("--write-freq", default=100)
    args = parser.parse_args()
    test_bicubic(lr_imgs=args.lr, hr_imgs=args.hr, output_file=Path(args.out), save_imgs=Path(args.save_img))


def test_bicubic(lr_imgs: Path, hr_imgs: Path, output_file: Path, save_imgs: Path, write_freq: int = 100, factor=2): #factor was hardcoded


    '''
    # ----------------------------------------
    # initialize model
    # ----------------------------------------
    '''

    dataset_opt = defaultdict(lambda: None, **{
        "name": "test_dataset",
        "dataset_type": "sr",
        "dataroot_H": str(hr_imgs),
        "dataroot_L": str(lr_imgs),
        "n_channels": 1
    })

    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    logging.debug("Forward-passing images")
    metrics = []

    if not Path(output_file).parent.exists():
        os.makedirs(str(Path(output_file).parent))

    if save_imgs and not save_imgs.exists():
        os.makedirs(str(save_imgs))

    for n, test_data in enumerate(tqdm(test_loader)):
        #l_img = cv2.imread(test_data['L_path'][0], cv2.IMREAD_GRAYSCALE)
        #h_img = cv2.imread(test_data['H_path'][0], cv2.IMREAD_GRAYSCALE)
        #e_img = cv2.resize(l_img, dsize=(l_img.shape[0]*4, l_img.shape[1]*4), interpolation=cv2.INTER_CUBIC).astype('uint8')
        l_img = ImageOps.grayscale(Image.open(test_data['L_path'][0]))
        h_img = np.array(ImageOps.grayscale(Image.open(test_data['H_path'][0])))
        e_img = np.array(l_img.resize((int(l_img.size[0] * factor), int(l_img.size[1] * factor)), resample=Image.BICUBIC))

        current_entry = {"L_path": test_data['L_path'][0], "H_path": test_data['H_path'][0], **get_metrics(e_img, h_img)}
        if save_imgs:
            img_path = str(save_imgs/f"{Path(test_data['L_path'][0]).stem}SwinIR{Path(test_data['L_path'][0]).suffix}")
            util.imsave(e_img, img_path=img_path)
            current_entry['O_path'] = img_path
        metrics.append(current_entry)

        if (n + 1) % write_freq == 0:
            df = pandas.DataFrame(metrics)
            df.to_csv(output_file, mode='a', header=not output_file.exists(), index=False)
            metrics = []

    df = pandas.DataFrame(metrics)
    df.to_csv(output_file, mode='a', header=not output_file.exists(), index=False)




if __name__ == '__main__':
    main()
