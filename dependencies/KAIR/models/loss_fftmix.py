# Frequency loss
# https://github.com/lj1995-computer-vision/Trident-Dehazing-Network/blob/master/loss/fft.py
import numpy as np
import torch
from torch.autograd import Variable


class FFTMixLoss(torch.nn.Module):
    def __init__(self, loss_f = torch.nn.L1Loss, reduction='mean', alpha:float =.3):
        super(FFTMixLoss, self).__init__()
        self.criterion = loss_f(reduction=reduction)
        assert 0 <= alpha <= 1
        self.alpha = alpha

    def forward(self, img1, img2):
        zeros=torch.zeros(img1.size()).to(img1.device)
        fft_loss = self.criterion(torch.fft.rfft(torch.stack((img1,zeros),-1),2),torch.fft.rfft(torch.stack((img2,zeros),-1),2))
        criterion_loss = self.criterion(img1, img2)
        loss = self.alpha * fft_loss + (1-self.alpha) * criterion_loss
        return loss


if __name__ == '__main__':
    import cv2
    from torch import optim
    from skimage import io
    npImg1 = cv2.imread("einstein.png")

    img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
    img2 = torch.rand(img1.size())

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=True)

    ssim_value = 1
    #print("Initial ssim:", ssim_value)

    ssim_loss = FFTMixLoss(alpha=1.0)
    optimizer = optim.Adam([img2], lr=0.01)

    while ssim_value > 0.01:
        optimizer.zero_grad()
        ssim_out = ssim_loss(img1, img2)
        ssim_value = ssim_out.item()
        print('{:<4.4f}'.format(ssim_value))
        ssim_out.backward()
        optimizer.step()
    img = np.transpose(img2.detach().cpu().squeeze().float().numpy(), (1,2,0))
    io.imshow(np.uint8(np.clip(img*255, 0, 255)))
