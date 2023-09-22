import argparse
import logging
import os
import pathlib
import re
import shutil
import sys
from glob import glob
from os import PathLike
import random
from typing import Optional, List

import PIL.Image
import numpy as np
from tqdm import tqdm
import yaml
from PIL import Image

from mikrosr.validation.matching import get_resolution, get_resolution_str


def getMX(match_str) -> Optional[float]:
    match_str = str(match_str)
    return res if (res := re.match(r'(\d+(.\d+)?)MX', match_str)) is None else float(res.group(1))

#import pydevd_pycharm
#pydevd_pycharm.settrace('134.169.30.95', port=56789, stdoutToServer=True, stderrToServer=True)


def split_data(folder_to_split: pathlib.Path, output_path: pathlib.Path, train_ratio: float, val_ratio: float,
               test_ratio: float, seed=None):
    """
    folder_to_split : path
    train_size, val_size, test_size : float values-> percentages

    """
    if seed:
        np.random.seed(seed)
    files = glob(str(folder_to_split / "*"), )
    random.shuffle(files)

    # ensure at least 1 test and 1 val img
    split_train_idx = min(int(train_ratio * len(files)), len(files) - 2)
    split_val_idx = split_train_idx + max(int(val_ratio * len(files)), 1)
    split_test_idx = split_val_idx + max(int(test_ratio * len(files)), 1)

    logging.debug(str(split_train_idx))
    logging.debug(str(split_val_idx))
    logging.debug(str(split_test_idx))

    # create folders
    os.makedirs(output_path / "train/source", exist_ok=True)
    os.makedirs(output_path / "val/source", exist_ok=True)
    os.makedirs(output_path / "test/source", exist_ok=True)

    # split
    files_train = files[:split_train_idx]
    files_val = files[split_train_idx:split_val_idx]  # bug?
    files_test = files[split_val_idx:split_test_idx]

    # copy to folders
    for image in files_train:
        shutil.copy(image, output_path / "train/source")
    for image in files_val:
        shutil.copy(image, output_path / "val/source")
    for image in files_test:
        shutil.copy(image, output_path / "test/source")


def create_zoom_pyramid(in_path: pathlib.Path, out_path: pathlib.Path, org_res: float, start_res: float, end_res: float,
                        steps: int, digits=2, upscale=False, sr_factor=4):
    if not out_path.exists():
        os.makedirs(out_path)
    if upscale:
        #changed these to * from / ....because inverse behavior of nm scala
        end_res = end_res * sr_factor
        start_res = start_res * sr_factor
    for resolution in tqdm(np.linspace(end_res, start_res, steps)):
        ratio = 1/(resolution / org_res) # problem here with inverse behavior of nm -> ratio would be 15/12 -> flip the ratio
        for img_file in in_path.glob('*.png'):
            image = Image.open(img_file)
            image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)),
                                 resample=PIL.Image.NEAREST if ratio <= 1 else PIL.Image.BICUBIC)
            image.save(f"{out_path / img_file.stem}{'-bc-'if upscale else ''}R{resolution * ((1/sr_factor) if upscale else 1):.{digits}f}{img_file.suffix}")


def generate_crops(in_path: PathLike, out_path: PathLike, crop_size=256, crops_per_row=8,
                   original_res=5, bicubic_crops_per_row=8):
    """
    in_path : path to folder containing .pngs
    out_path : path to folder for output crops
    """
    in_path = pathlib.Path(in_path)
    out_path = pathlib.Path(out_path)

    os.makedirs(out_path, exist_ok=True)

    files = os.listdir(str(in_path))

    for file in tqdm(files):
        if file.endswith('.png'):

            image = np.asarray(Image.open(in_path / file))
            res = get_resolution(file)
            if res > original_res:
                current_crops_per_row = bicubic_crops_per_row
            else:
                current_crops_per_row = crops_per_row

            x = 0
            y = 0
            i = 0
            # logging.debug(image.shape)
            if image.shape[0] < crop_size or image.shape[1] < crop_size:
                logging.warning(f"Skipping {file}: Smaller than crop size!")
                continue
            if current_crops_per_row == 1:
                x_shift= image.shape[0]
                y_shift=image.shape[1]
            else:
                x_shift = max((image.shape[0] - crop_size) // (current_crops_per_row - 1), 1)  # at least 1
                y_shift = max((image.shape[1] - crop_size) // (current_crops_per_row - 1),
                          1)  # for shape==cropsize edge case
            # logging.debug(f"x shift: {x_shift}")
            while image.shape[1] - y >= crop_size:  # y loop
                # logging.debug(f"y={y}")
                while image.shape[0] - x >= crop_size:  # x loop
                    # logging.debug(f"x={x}")

                    crop = image[x:x + crop_size, y:y + crop_size]

                    img = Image.fromarray(crop)

                    img.save(out_path / f"{file[0:-4]}_crop{i}.png")
                    # logging.debug(str(out_path) + '/' + file + '_crop'+str(i) +".png  saved")

                    i += 1
                    x += x_shift

                y += y_shift
                x = 0

    return logging.debug("done")


def downsample_with_filters(in_path: PathLike, hr_path: PathLike, lr_path: PathLike, ratio: int,
                            filters=Optional[List[int]]):
    """

    uses a resample/downsample algorithm on all images in the in_path-folder and stores them in out_path/{resample}_ds{ratio}X

    in_path : path to folder containing .pngs
    out_path : path to folder for output crops
    ratio : the downsample ratio
    ds_type : the downsample algorithm used


    pillow downsampling used
    https://pillow.readthedocs.io/en/stable/reference/Image.html
    Image.resize(size,resample=None,box=None,reducing_gap=None)

    resample is the resampling filter:
    ["NEAREST", "LANCZOS", "BILINEAR", "BICUBIC", "BOX", "HAMMING"] -> 0,1,2,3,4,5

    with box we can chose a region to be resized

    Filters in Python Imaging Library
    https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
    """

    in_path = str(in_path)
    target_path = str(lr_path)
    hr_path = str(hr_path)

    files = os.listdir(in_path)
    filter_names = ["NEAREST", "LANCZOS", "BILINEAR", "BICUBIC", "BOX", "HAMMING"]
    filters = filters or range(0, 6)

    os.makedirs(target_path, exist_ok=True)

    HR_path = hr_path
    os.makedirs(HR_path, exist_ok=True)

    for file in tqdm(files):
        if file.endswith('.png'):

            image = Image.open(in_path + "/" + file)

            for filter in filters:
                img_name = file[0:-4] + '_' + str(filter)
                if ratio>1:
                    image.save(HR_path + '/' + img_name + '.png')
                    image_lr = image.resize((int(image.size[0] / ratio), int(image.size[1] / ratio)), resample=filter)
                    image_lr.save(target_path + '/' + img_name + 'X' + str(ratio) + '.png')
                else:
                    image_hr = image.resize((int(image.size[0] / ratio), int(image.size[1] / ratio)), resample=filter)
                    image_hr.save(HR_path + '/' + img_name + '.png')
                    image.save(target_path + '/' + img_name + 'X' + str(int(1/ratio)) + '.png')


def create_image_pair_subset(l_in_path: pathlib.Path, h_in_path: pathlib.Path, l_out_path: pathlib.Path,
                             h_out_path: pathlib.Path, n: int, ratio: int):
    os.makedirs(l_out_path, exist_ok=True)
    os.makedirs(h_out_path, exist_ok=True)
    h_files = h_in_path.glob("*")
    img_pairs = [(hf, l_in_path / f"{hf.stem}X{ratio}{hf.suffix}") for hf in h_files]
    sampled_pairs = random.sample(img_pairs, n)
    for h_img, l_img in tqdm(sampled_pairs):
        shutil.copy(h_img, h_out_path / h_img.name)
        shutil.copy(l_img, l_out_path / l_img.name)


def create_dataset(config: PathLike, outpath: PathLike, overwrite: bool = False):
    config = pathlib.Path(config)
    output_root = pathlib.Path(outpath)
    logging.info(f"Creating dataset from {config}")

    with open(config, "r") as f:
        try:
            options = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.error(f"Error while parsing {config.name}")
            sys.exit(1)

    if seed := options.get('seed'):
        logging.info(f"Setting random seed {seed}")
        random.seed(seed)

    if overwrite and output_root.exists():
        logging.info('The --overwrite flag is set, deleting existing dataset')
        shutil.rmtree(output_root)
    try:
        os.makedirs(output_root)
    except OSError:
        logging.error(f"outpath {outpath} already exist. Please specify a non-existing directory or use --overwrite"
                      f"flag to delete before creation.")
        sys.exit(1)
    shutil.copy(config, output_root / config.name)

    source_dir = output_root / "source"
    os.makedirs(source_dir, exist_ok=True)
    shutil.copytree(options['set']['inpath'], source_dir, dirs_exist_ok=True)

    logging.info(f"Copied source images and options.")

    split_sets = ['train', 'val', 'test']

    logging.info(f"Splitting dataset...")
    split_data(source_dir, output_root,
               options['split']['train']['ratio'],
               options['split']['val']['ratio'],
               options['split']['test']['ratio'])
    for split_set in split_sets:
        logging.info(f"-- Preparing {split_set} set --")
        split_set_options = options['split'][split_set]
        if split_set_options.get('skip'):
            logging.info(f"Skipping split {split_set}")
            continue

        split_root = output_root / split_set
        split_source = split_root / "source"
        pyramid_dir = split_root / "pyramid"
        bc_pyramid_dir = split_root / "bc_pyramid"
        crops_dir = split_root / "crops"
        bc_crops_dir = split_root / "bc_crops"
        hr_dir = split_root / "HR"
        lr_dir = split_root / "LR"

        tuple(map(lambda x: os.makedirs(x, exist_ok=True), [
            split_source,
            bc_pyramid_dir,
            pyramid_dir,
            crops_dir,
            bc_crops_dir,
            hr_dir,
            lr_dir
        ]))

        if add_imgs := split_set_options.get("additional_images"):
            logging.info("Copying additional after-split source images...")
            shutil.copytree(add_imgs, split_source, dirs_exist_ok=True)

        downsampling_factor = options['crops'].get('downsample_with_factor')

        logging.info("Creating zoom pyramid...")
        digits = split_set_options['res_digits']
        for res in split_set_options["target_res"]:
            create_zoom_pyramid(
                in_path=split_source,
                out_path=pyramid_dir,
                org_res=options['set']['original_res'],
                start_res=res[1], end_res=res[0], steps=res[2],
                digits=digits,
                upscale=False,
                sr_factor=downsampling_factor or 4
            )
        if bc_target_res := split_set_options.get("bc_target_res"):
            logging.info("Creating bc zoom pyramid...")
            for res in bc_target_res:
                create_zoom_pyramid(
                    in_path=split_source,
                    out_path=bc_pyramid_dir,
                    org_res=options['set']['original_res'],
                    start_res=res[1], end_res=res[0], steps=res[2],
                    digits=digits,
                    upscale=True,
                    sr_factor=downsampling_factor or 4
                )

        logging.info("Cutting Crops...")
        crops_per_row = split_set_options.get('crops_per_row_col') or options['crops']['crops_per_row_col']
        generate_crops(
            in_path=pyramid_dir,
            out_path=crops_dir,
            crop_size=options['crops']['size'],
            crops_per_row=crops_per_row,
            original_res=options['set']['original_res'],
            bicubic_crops_per_row=(
                bc_crops_per_row := options.get('bicubic', {}).get('crops_per_row_col') or crops_per_row)
        )
        if next(bc_pyramid_dir.iterdir(), None):  # test for not-empty directory
            logging.info("Cutting bc Crops...")
            generate_crops(
                in_path=bc_pyramid_dir,
                out_path=bc_crops_dir,
                crop_size=options['crops']['size'] // downsampling_factor,
                crops_per_row=options.get('bicubic', {}).get('crops_per_row_col_upscale') or crops_per_row,
                original_res=options['set']['original_res'],
                bicubic_crops_per_row=options.get('bicubic', {}).get('crops_per_row_col_upscale') or crops_per_row
            )

        if downsampling_factor \
                and not options['crops'].get('skip_downsampling'):
            logging.info("Downsampling...")
            downsample_with_filters(
                in_path=crops_dir,
                hr_path=hr_dir,
                lr_path=lr_dir,
                ratio=downsampling_factor,
                filters=split_set_options.get('filters')
            )
            if next(bc_crops_dir.iterdir(), None):  # test for not-empty directory
                logging.info("bc upsampling...")
                downsample_with_filters(
                    in_path=bc_crops_dir,
                    hr_path=hr_dir,
                    lr_path=lr_dir,
                    ratio=1 / downsampling_factor,
                    filters=split_set_options.get("upsample_filters", [3])
                )

        if sample_size := split_set_options.get('generate_subset'):
            logging.info("Generating subset...")
            if downsampling_factor:
                subset_lr_dir = split_root / "LR_subset"
                subset_hr_dir = split_root / "HR_subset"
                os.makedirs(subset_lr_dir)
                os.makedirs(subset_hr_dir)

                create_image_pair_subset(
                    l_in_path=lr_dir,
                    h_in_path=hr_dir,
                    l_out_path=subset_lr_dir,
                    h_out_path=subset_hr_dir,
                    n=sample_size,
                    ratio=downsampling_factor
                )
            else:
                subset_dir = split_root / "crops_subset"
                os.makedirs(subset_dir)
                files = list(crops_dir.glob('*'))
                subset_files = random.sample(files, sample_size)
                for file in tqdm(subset_files):
                    shutil.copy(file, subset_dir)


def main(args=None):
    parser = argparse.ArgumentParser("Create a microsr from .yml file")

    parser.add_argument("configfile", help=".yml config file for dataset")
    parser.add_argument("outpath", help="path to write dataset to")
    parser.add_argument("--overwrite", help="Delete outpath before creation if already existing", action="store_true")

    args = parser.parse_args(args)

    create_dataset(pathlib.Path(args.configfile), pathlib.Path(args.outpath), args.overwrite)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO)
    main()
