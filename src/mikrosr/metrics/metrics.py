import math

import cv2
import lpips
import numpy as np
import torch
from sewar.full_ref import ssim, msssim
from pytorch_msssim.ssim import ms_ssim

from mikrosr.metrics.defects import GaussianDefect
from utils.utils_image import calculate_psnr

"""
This module contains all implementations of the metrics used within the project
"""

class Metric:
    """
    An abstract base class with just the core functionalities
    """
    def __init__(self):
        self.name = "abstract metric"
        self.unit = ""

    def measure(self, img1, img2):
        pass


class LPIPS(Metric):
    """
    LPIPS wrapper using https://github.com/richzhang/PerceptualSimilarity
    """
    def __init__(self):
        super().__init__()
        self.name = "LPIPS"
        self.unit = ""
        self.loss_fn_alex = lpips.LPIPS(net='alex')

    def measure(self, img1, img2):
        img1_float = img1.astype(np.double)
        img2_float = img2.astype(np.double)
        img1_float = 2*img1_float/255 - 1
        img2_float = 2*img2_float/255 - 1
        t1 = torch.from_numpy(img1_float).float()
        t2 = torch.from_numpy(img2_float).float()
        if len(img1.shape) > 2:
            t1 = torch.unsqueeze(torch.transpose(torch.transpose(t1, 0, 2), 1, 2), 0)
            t2 = torch.unsqueeze(torch.transpose(torch.transpose(t2, 0, 2), 1, 2), 0)
        else:
            t1 = torch.unsqueeze(torch.unsqueeze(t1, 0), 0).repeat(1, 3, 1, 1)
            t2 = torch.unsqueeze(torch.unsqueeze(t2, 0), 0).repeat(1, 3, 1, 1)

        d = self.loss_fn_alex(t1, t2)
        return float(torch.squeeze(d))



class PSNR(Metric):
    """
    PSNR wrapper for the KAIR implementation
    """
    def __init__(self):
        super().__init__()
        self.name= "PSNR (dB)"
        self.unit = "dB"

    def measure(self, img1, img2):
        print("hello"+str(img1.shape) +"_"+ str(img2.shape))
        return calculate_psnr(img1, img2)


class SSIM(Metric):
    """
    SSIM from the image-similarity-measures package https://pypi.org/project/image-similarity-measures/
    https://github.com/andrewekhalel/sewar/blob/master/sewar/full_ref.py
    """
    def __init__(self):
        super().__init__()
        self.name = "SSIM"

    def measure(self, img1, img2):
        return ssim(img1, img2)[0]


class MS_SSIM(Metric):
    """
    Wrapper for implementation by https://github.com/jorge-pessoa/pytorch-msssim.
    """
    def __init__(self):
        super().__init__()
        self.name = "MS-SSIM"

    def measure(self, img1:np.ndarray, img2:np.ndarray):
        # np array to pytorch tensor conversion
        img1_int = img1.astype(float)
        img2_int = img2.astype(float)
        t1 = torch.from_numpy(img1_int)
        t2 = torch.from_numpy(img2_int)
        if len(img1.shape) > 2:
            t1 = torch.unsqueeze(torch.transpose(torch.transpose(t1, 0, 2), 1,2),0)
            t2 = torch.unsqueeze(torch.transpose(torch.transpose(t2, 0, 2), 1, 2),0)
        else:
            t1 = torch.unsqueeze(torch.unsqueeze(t1,0),0)
            t2 = torch.unsqueeze(torch.unsqueeze(t2, 0), 0)

        return float(ms_ssim(t1, t2))


class GaussMetric(Metric):
    """
    A decorator class to convert a metric into a Gauss-Metric as described in Sec. 2.2
    """
    def __init__(self, metric: Metric):
        super(GaussMetric, self).__init__()
        self.metric = metric
        self.name = f"Gauss metric ({self.metric.name})"
        self.unit = self.metric.unit

    def measure(self, img1, img2):
        stack1, stack2 = GaussMetric.get_stack(img1), GaussMetric.get_stack(img2)
        return np.average([self.metric.measure(i1, i2) for i1, i2 in zip(stack1, stack2)])

    @staticmethod
    def get_stack(img):
        return np.array((img, cv2.GaussianBlur(img, (3,3), 0), cv2.GaussianBlur(img, (5,5), 0)))


_eval_metrics = None


def get_evaluation_metrics():
    global _eval_metrics
    if not _eval_metrics:
        _eval_metrics = [PSNR(), SSIM(), MS_SSIM(), GaussMetric(PSNR()), GaussMetric(SSIM()), GaussMetric(MS_SSIM())]#, LPIPS()]
    return _eval_metrics




def main():
    # Some test code for comparing an image with a noisy variant of itself

    img = cv2.imread(
        "/home/jan/hdd/PycharmProjects/projektarbeitsr/data/STEM/5MX_png/gray/BS3116_1b_L1_ADF_5MX_2715_gray_png.png")
    sigma = 3
    gd = GaussianDefect(sigma)
    dist_img = gd.disturb(img)
    m1 = PSNR()
    m2 = GaussMetric(PSNR())

    print("test")
    print(f"same image: {m1.name} = {m1.measure(img, img)}")
    print(f"same image: {m2.name} = {m2.measure(img, img)}")

    print(f"with noise (sigma = {sigma}): {m1.name} = {m1.measure(img, gd.disturb(img))}")
    print(f"with noise (sigma = {sigma}): {m2.name} = {m2.measure(img, gd.disturb(img))}")


if __name__ == '__main__':
    main()



