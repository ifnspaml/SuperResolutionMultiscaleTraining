# %%
import argparse
import logging
import os

from mikrosr.metrics import metrics
# %%
import itertools
import re
from decimal import Decimal
from os import PathLike
from pathlib import Path as P
from typing import Optional

from tqdm import tqdm

import cv2
import pandas as pd

from mikrosr.validation.forward_pass import forward_pass, forward_pass_bc
from mikrosr.validation.matching_nm import MatchingLoader
import mikrosr.validation.matching_nm as matching
from mikrosr.metrics import metrics as metrics

# %%
from mikrosr.validation.test_model_on_set import get_model_option

# %%
def matching_evaluate(loader: MatchingLoader, output: PathLike, save_matches: Optional[PathLike],
                      write_freq: int = 100):
    results = []
    current_line = {}
    output = P(output)

    if not output.parent.exists():
        os.makedirs(output.parent)

    save_matches = P(save_matches) if save_matches else None
    if save_matches and not save_matches.exists():
        os.makedirs(str(save_matches))

    for n, ((c_path, crop), (m_path, match_img)) in tqdm(enumerate(loader.get_pairs()), total=len(loader)):
        current_line.update({'c_path': c_path, 'm_path': m_path})
        match_crop = matching.rot_scale_align(crop=crop, img_full=match_img, metric=metrics.PSNR())['matched_crop']
        metric_results = matching.get_metrics(crop, match_crop)
        current_line.update(metric_results)
        if save_matches:
            mc_path = save_matches / f"{c_path.stem}_match{c_path.suffix}"
            cv2.imwrite(str(mc_path), match_crop)
            current_line.update({'mc_path': mc_path})
        results.append(current_line)
        current_line = {}

        if (n + 1) % write_freq == 0:
            df = pd.DataFrame(results)
            df.to_csv(output, mode='a', header=not output.exists(), index=False)
            results = []
    df = pd.DataFrame(results)
    df.to_csv(output, mode='a', header=not output.exists(), index=False)


def main(args=None):
    parser = argparse.ArgumentParser("Template-match LR-Crops against zoom pyramid")
    parser.add_argument("--model", type=P, help="path to model directory to evaluate")
    parser.add_argument("--crops", type=P, help="64x64 Crops from lower zoomlevel")
    parser.add_argument("--zp", type=P, help="zoom pyramid to match against")
    parser.add_argument("--out", type=P, help="output directory")
    parser.add_argument("--baseline", action="store_true",
                        help="upsample crops with bc interpolation instead of model")
    parser.add_argument("--factor",type=float, help="SR factor")

    args = parser.parse_args(args)

    out: P = args.out
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)

    fp_dir = out / "forwardpass"
    csv_out = out / "results.csv"
    matches_out = out/"matches"
    if args.baseline:
        logging.info("Bicubic interpolating crops...")
        forward_pass_bc(input_path=args.crops, output_path=fp_dir)
    else:
        model_pth, opt_pth = get_model_option(args.model)

        logging.info("Forward-passing crops...")
        forward_pass(model_path=model_pth, opt_path=opt_pth, input_path=args.crops, output_path=fp_dir)

    loader = MatchingLoader(fp_dir, args.zp, factor=args.factor)
    logging.info("matching crops...")
    matching_evaluate(loader=loader, output=csv_out, save_matches=matches_out)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
