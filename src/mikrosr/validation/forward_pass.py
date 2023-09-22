import argparse
import glob
import itertools
import logging
import os.path
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import ImageOps, Image

from tqdm import tqdm

from mikrosr.validation.test_model_on_set import get_model_option
from models.select_network import define_G
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#import pydevd_pycharm
#pydevd_pycharm.settrace('134.169.30.95', port=56789, stdoutToServer=True, stderrToServer=True)

'''
This is a modified version of the SwinIR test code for forward passing unpaired images through a swinIR model


Original source of code:
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def forward_pass_bc(input_path: Path, output_path: Path):
    if not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    for file in tqdm(list(itertools.chain(input_path.glob('*.png'),input_path.glob('*.jpg')))):
        l_img = ImageOps.grayscale(Image.open(file))
        e_img = np.array(l_img.resize((int(l_img.size[0] * 2), int(l_img.size[1] * 2)), resample=Image.BICUBIC))
        util.imsave(e_img, str(output_path/f"{file.stem}_bc{file.suffix}"))


def forward_pass(model_path, input_path, output_path, opt_path=Optional[Path]):
    logging.debug("Parsing options")
    logging.debug(f"Output path: {output_path}")
    model_path, opt_path = get_model_option(model_path, opt_path)
    opt = option.parse(opt_path, is_train=True)
    os.makedirs(output_path, exist_ok=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------

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
    logging.debug("Init model")
    model = define_G(opt)
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model)
    model.to(device)
    model.eval()

    window_size = opt['netG']['window_size']
    scale = opt['scale']

    # -------------------------------
    #  testing
    # -------------------------------
    logging.debug("Forward-passing images")
    for path in tqdm(sorted(glob.glob(os.path.join(input_path, '*')))):
        # read image
        (imgname, imgext) = os.path.splitext(os.path.basename(path))
        img_lq = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.  # image to HWC-BGR, float32
        # img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
        #                      (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = model(img_lq)
            output = output[..., :h_old * scale, :w_old * scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{output_path}/{imgname}_SwinIR.png', output)


def main(json_path='options/train_msrresnet_psnr.json'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option JSON file.')
    parser.add_argument('--model')
    parser.add_argument('--images')
    parser.add_argument('--output')
    args = parser.parse_args()
    forward_pass(opt_path=args.opt, model_path=args.model, input_path=args.images, output_path=args.output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
