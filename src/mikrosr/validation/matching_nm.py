import argparse
import logging
import os.path
import re
from glob import glob
from os import PathLike
from typing import List, Callable

import cv2
import numpy as np
import pandas as pd

import mikrosr.metrics.metrics as metrics
from mikrosr.metrics.metrics import get_evaluation_metrics
from mikrosr.progress import Progress
from pathlib import Path as P

""" This file contains the functions used to realize the custom template matching algorithm described in chapter 4"""

eval_metrics = get_evaluation_metrics()


def get_metrics(img1, img2):
    return {m.name: m.measure(img1, img2) for m in eval_metrics}


# https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def alignFull(img_full, crop, do_blur=True):
    """
    Find a matching patch via simple template matching. Optional gaussian denoising
    :return: (topleft, bottomright), matched_crop
    """
    if do_blur:
        img_full_blur = cv2.blur(img_full, (3, 3))
        crop = cv2.blur(crop, (3, 3))
    else:
        img_full_blur = img_full
    res = cv2.matchTemplate(img_full_blur, crop, cv2.TM_CCOEFF)
    _, _, max_val, max_loc = cv2.minMaxLoc(res)
    bottom_right = np.array(max_loc) + np.array(crop.shape)
    return (max_loc, bottom_right), img_full[max_loc[1]:bottom_right[1], max_loc[0]:bottom_right[0]]


def align(img_full, crop, do_blur=True):
    """
    like alignFull
    :return: only the matched patch

    """
    return alignFull(img_full, crop, do_blur)[1]


def alignPosition(img_full, crop, do_blur=True):
    """
    like alignFull
    :return: only top left and bottom right
    """
    return alignFull(img_full, crop, do_blur)[0]


def optimize(img_full: np.ndarray, crop: np.ndarray, values: List[float], transformation: Callable,
             metric: metrics.Metric, do_blur=True):
    """
    Optimizes a transformation along values with respect to the best match of crop agaings img_full,
    measured with metric
    """
    metric_results = []
    for value in values:
        transformed_img = transformation(img_full, value)
        metric_results.append(
            (value, transformed_img, metric.measure(crop, align(transformed_img, crop, do_blur=do_blur))))
    return max(metric_results, key=lambda x: x[2])


def optimize_scale(img_full, crop, range, samples, metric, do_blur=True):
    """
    Scale version of optimize
    """
    values = np.linspace(1 - range, 1 + range, samples)
    transformation = lambda img_full, value: cv2.resize(img_full, None, fx=value, fy=value)
    return optimize(img_full, crop, values, transformation, metric, do_blur=do_blur)


def optimize_rotation(img_full, crop, range, samples, metric, do_blur=True):
    """
        Rotation version of optimize
    """
    values = np.linspace(-range, range, samples)
    pos = alignPosition(img_full, crop, do_blur=do_blur)
    center = [int(sum(p) / 2) for p in zip(*pos)]
    transformation = lambda img_full, angle: rotate_image(img_full, angle)
    return optimize(img_full, crop, values, transformation, metric, do_blur=do_blur)


def get_perimeter(img, max_loc, bottom_right, padding):
    """
    Cut out a rectangle spanned by max_loc and bottom_right plus padding out of an image
    """
    h, w = img.shape
    return img[
           max(0, max_loc[1] - padding):min(h, bottom_right[1] + padding),
           max(0, max_loc[0] - padding):min(w, bottom_right[0] + padding)
           ]


def rot_scale_align(img_full, crop, do_blur=True, samples=25, padding=50, scale_range=0.1, rot_range=3,
                    metric=metrics.SSIM()):
    """
    This function runs the algorithm described in Chapter 4, except the metrics are not calculated. Only the matched crop
    and the found scale and rotation are returned
    :param img_full: The larger image to match agaings
    :type img_full: np.ndarray, shape (H, W)
    :param crop: the original crop to match within the image
    :type crop: np.ndarray, shape (H, W)
    :param do_blur: Use a 3x3 gauss filter before doing each template matching
    :type do_blur: bool
    :param samples: Number of sampes for the rotation and scaling linspaces
    :type samples: int
    :param padding: Number of pixels padding at each side of the perimeter after first template matching
    :type padding: int
    :param scale_range: +- scale_range is sampled for scaling factors to find the optimal scale.
    :type scale_range: float
    :param rot_range: +- rot_range is sampled for rotation angles factors to find the optimal scale.
    :type rot_range: float, rotation in degrees
    :param metric: The metric used to determine the optimal scale/rotation
    :type metric: metrics.Metric
    :return: a dict containing found scale, rotation, and the matched crop
    :rtype: dict
    """
    (max_loc, bottom_right), matched_crop = alignFull(img_full, crop, do_blur=do_blur)
    match_perimeter = get_perimeter(img_full, max_loc, bottom_right, padding=padding)

    if matched_crop.shape[0]*1.15 <= matched_crop.shape[0] or matched_crop.shape[1]*1.15 <= matched_crop.shape[1]:


        #for STM data we run into the problem, that crop and image to be matched are of the same pixel resolution.
        #hot fix for now: Skip Scale and Rotate for those.
        #TODO: Cut out middle part of crop e.g. 200px x 200px and use it for comparison

        scale, scaled_img, scale_ssim = optimize_scale(match_perimeter, crop, range=scale_range, samples=samples,
                                                       metric=metric)
        rotation, rot_img, rot_ssim = optimize_rotation(scaled_img, crop, range=rot_range, samples=samples, metric=metric)

        return dict(scale=scale, rotation=rotation, matched_crop=align(rot_img, crop, do_blur=True))
    else:
        return dict(scale=1, roration=1, matched_crop=matched_crop)




def cross_match(crops, images, method=cv2.TM_CCORR, **kwargs):
    """
    A helper function for matching multiple crops against multiple images and calculating the similarity metrics.
    All parameters for rot_scale_align can be given in kwargs.
    The result is returned as a dataframe, containing metric results, original crops and source crops.
    """
    results = []
    counter = 0
    prog = Progress(len(crops) * len(images), text="matching crops", print_every=1)
    prog.start_time()
    prog.print_progress(0)
    for crop in crops:
        for image in images:
            matched_crop = rot_scale_align(image, crop, **kwargs)['matched_crop']
            current_result = get_metrics(crop, matched_crop)
            logging.info(current_result)
            results.append({'crop': crop, 'match': matched_crop, **current_result})

            """plt.subplot(131), plt.imshow(crop, cmap='gray_r', vmax=255, vmin=0)
            plt.title('original crop')
            plt.subplot(132), plt.imshow(matched_crop, cmap='gray_r', vmax=255, vmin=0)
            plt.title('matched crop')
            plt.subplot(133), plt.imshow(np.abs(matched_crop - crop), cmap='gray_r', vmax=255, vmin=0)
            plt.title('difference')
            plt.show()"""

            counter += 1
            prog.print_progress(counter)
    # avg = {k: results[k]/counter for k in results.keys()}
    prog.done()
    return pd.DataFrame.from_dict(results)


def load_and_cosstest(crops_path, images_path, do_blur=True):
    # Helper function for crossmatching two directories and calculating the average
    crops = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob(f"{crops_path}/*") if os.path.isfile(f)]
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob(f"{images_path}/*") if os.path.isfile(f)]
    avg = cross_match(crops, images, do_blur=do_blur)
    return avg


def get_resolution(path:PathLike):
    return float(get_resolution_str(path))


def get_resolution_str(path:PathLike):
    path = P(path)
    return re.search(r"R(\d+(.\d+)?)", str(path.name)).group(1)


class MatchingLoader:
    def __init__(self, crop_path:PathLike, matching_path:PathLike, factor:int, digits=2):
        self.factor = factor
        self.crop_path = P(crop_path)
        self.matching_path = P(matching_path)
        self.digits = digits

        #change /factor to to *factor - since it is reversed for nm
        self.resolution_map = {f"{get_resolution(f)*factor:.{self.digits}f}": f for f in  self.matching_path.glob('*')}

    def __len__(self):
        return len(list(self.crop_path.glob('*')))

    def get_pairs(self):
        cached_match_image = None
        current_resolution = None
        for crop in self.crop_path.glob('*'):
            res = f"{get_resolution(crop):.{self.digits}f}"
            #TODO: remove this hack
            if float(res)<0.94: continue
            if not res == current_resolution:
                current_resolution = res
                matching_imag_path = self.resolution_map[current_resolution]
                cached_match_image = cv2.imread(str(matching_imag_path), cv2.IMREAD_GRAYSCALE)
            crop_img = cv2.imread(str(crop), cv2.IMREAD_GRAYSCALE)
            yield (crop, crop_img), (matching_imag_path, cached_match_image)


def main():
    # Driver code to run the crossmatching as main
    parser = argparse.ArgumentParser("Crosstest crops against images")
    parser.add_argument("--crops")
    parser.add_argument("--images")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(load_and_cosstest(args.crops, args.images))


if __name__ == '__main__':
    main()
