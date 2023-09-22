import argparse
import glob
import logging
import os.path
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import pandas
import torch

from pathlib import Path
from tqdm import tqdm
from models.select_model import define_Model
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from torch.utils.data import DataLoader

from mikrosr.validation.matching import get_metrics
from mikrosr.validation.test_bicubic_on_set import test_bicubic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# print("starting debug")
# import pydevd_pycharm
# pydevd_pycharm.settrace('134.169.30.95', port=8223, stdoutToServer=True, stderrToServer=True)
# print("after debug")


def get_model_option(model_path: Optional[Path], option_path: Optional[Path] = None):
    model_out, option_out = [None] * 2
    if model_path and model_path.is_dir():
        model_out = next(model_path.glob('*_E.pth'), None)
        if not option_path:
            option_out = next(model_path.glob('*.json'), None)
    return model_out or model_path, option_out or option_path


def main():
    parser = argparse.ArgumentParser("Evaluate a model on a testset")
    parser.add_argument("--opt", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--lr", type=Path)
    parser.add_argument("--hr", type=Path)
    parser.add_argument("--set", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--factor", default=2, type=int)
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--write-freq", default=100)
    args = parser.parse_args()
    out = Path(args.out)

    args.lr = args.lr or args.set / "LR"
    args.hr = args.hr or args.set / "HR"
    args.model, args.opt = get_model_option(args.model, args.opt)
    if args.model:
        test_model(opt_path=args.opt, model_path=args.model, lr_imgs=args.lr, hr_imgs=args.hr,
                   output_file=out / "model.csv", save_imgs=out / "output_model" if args.save_img else None)
    if args.baseline:
        test_bicubic(lr_imgs=args.lr, hr_imgs=args.hr, output_file=out / "baseline.csv",
                     save_imgs=out / "output_baseline" if args.save_img else None, factor=args.factor or 4)


class Evaluator:
    def evaluate(self, lr, hr):
        pass


class ModelEvaluator:
    def __int__(self, opt_path: Path, model_path: Path):
        opt = option.parse(opt_path, is_train=False)

        # ----------------------------------------
        # return None for missing key
        # ----------------------------------------
        opt = option.dict_to_nonedict(opt)

        # ----------------------------------------
        # configure logger
        # ----------------------------------------
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

        '''
        # ----------------------------------------
        # initialize model
        # ----------------------------------------
        '''
        opt['E_decay'] = 1
        opt['path']['pretrained_netG'] = model_path
        opt['path']['pretrained_netE'] = model_path
        logging.debug("Init model")
        model = define_Model(opt)
        model.load()


def test_model(opt_path: Path, model_path: Path, lr_imgs: Path, hr_imgs: Path, output_file: Path, save_imgs: Path,
               write_freq: int = 100):
    if save_imgs:
        os.makedirs(save_imgs, exist_ok=True)

    os.makedirs(output_file.parent, exist_ok=True)
    logging.debug("Parsing options")
    opt = option.parse(opt_path, is_train=False)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    '''
    # ----------------------------------------
    # initialize model
    # ----------------------------------------
    '''
    opt['E_decay'] = 1
    opt['path']['pretrained_netG'] = str(model_path)
    logging.debug("Init model")
    model = define_Model(opt)
    model.load()

    window_size = opt['netG']['window_size']
    scale = opt['scale']

    dataset_opt = defaultdict(lambda: None, **{
        "name": "test_dataset",
        "dataset_type": "sr",
        "dataroot_H": str(hr_imgs),
        "dataroot_L": str(lr_imgs),
        "n_channels": 1
    })

    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=0,
                             drop_last=False, pin_memory=True)

    logging.debug("Forward-passing images")
    metrics = []
    for n, test_data in enumerate(tqdm(test_loader)):
        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        e_img = util.tensor2uint(visuals['E'])
        h_img = util.tensor2uint(visuals['H'])

        current_entry = {"L_path": test_data['L_path'][0], "H_path": test_data['H_path'][0],
                         **get_metrics(e_img, h_img)}
        if save_imgs:
            img_path = str(
                save_imgs / f"{Path(test_data['L_path'][0]).stem}SwinIR{Path(test_data['L_path'][0]).suffix}")
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
    logging.basicConfig(level="INFO")
    main()
