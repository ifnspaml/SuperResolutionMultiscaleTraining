import argparse
import os
from pathlib import Path as P

import cv2
import numpy as np
from tqdm import tqdm

from mikrosr.stm.read import read


def flatten(data, order=1, mask=[]):
    '''
    Taken from NSFOpen.example.Plotting_Data by nelson<at>nanosurf.com
    '''
    data_out = np.copy(data)  # create copy of data
    data_in = np.copy(data)

    data_in = data_in.astype(float)
    if np.any(mask):
        data_in[mask] = np.nan
    for idx, (out, line) in enumerate(zip(data_out, data_in)):
        ix = np.isfinite(line)

        x = np.arange(len(line))
        p = np.polyfit(x[ix], line[ix], order)  # fit data to polynomial
        y = np.polyval(p, x)
        data_out[idx] = out - y  # subtract fit from data
    return data_out


def main(args=None):
    parser = argparse.ArgumentParser("Preprocess .nid images")

    parser.add_argument('indir', help="input directory containing .nid files for preprocessing", type=P)
    parser.add_argument('outdir', help="output directory for preprocessed .png files", type=P)
    parser.add_argument('--threshold', '-t', help="cutoff percent for outliers", type=float, default=2)
    parser.add_argument('--fitorder', '-f', help="order of polynomial fit for flattening", type=int, default=1)

    args = parser.parse_args(args)

    if not args.outdir.exists():
        os.makedirs(args.outdir)

    files = list(args.indir.glob('**/*.nid', ))
    for file in tqdm(files):
        stm_file = read(file)
        datasets = {
            'forward': stm_file.data['Image']['Forward']['Z-Axis'] * 1e9,  # Scale from m to nm
            'backward': stm_file.data['Image']['Backward']['Z-Axis'] * 1e9
        }
        outdir = args.outdir / file.relative_to(args.indir).parent
        if not outdir.exists():
            os.makedirs(outdir)

        for key, dataset in datasets.items():
            dataset = flatten(dataset)
            clip_min, clip_max = [np.percentile(dataset, percent) for percent in [args.threshold, 100 - args.threshold]]
            clip = np.clip(dataset, clip_min, clip_max)
            grayscale = ((clip - np.min(clip_min)) / (np.max(clip_max) - np.min(clip_min)) * 255).astype(np.uint8)
            cv2.imwrite(str(outdir/f"{file.stem}-{key}-t{args.threshold:1d}-f{args.fitorder:1d}.png"), grayscale)
            pass


if __name__ == '__main__':
    main()
