# Frequency loss
# https://github.com/lj1995-computer-vision/Trident-Dehazing-Network/blob/master/loss/fft.py
import torch


class FFTL1MixLoss(torch.nn.Module):
    def __init__(self, loss_f = torch.nn.L1Loss, reduction='mean', alpha=.3):
        super(FFTL1MixLoss, self).__init__()
        self.criterion = loss_f(reduction=reduction)
        assert 0 <= alpha <= 1
        self.alpha = alpha

    def forward(self, img1, img2):
        zeros=torch.zeros(img1.size()).to(img1.device)
        return self.criterion(torch.fft(torch.stack((img1,zeros),-1),2),torch.fft(torch.stack((img2,zeros),-1),2))

