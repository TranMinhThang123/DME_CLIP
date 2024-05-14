import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence
import kornia

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        valid_mask = (target > 0).detach()
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        input = input[valid_mask]
        target = target[valid_mask]
        
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        print("Silog: ",10*torch.sqrt(Dg))
        return 10 * torch.sqrt(Dg)


class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

class SSIMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ssim = kornia.losses.SSIMLoss(window_size=7,max_val=10.)
    def forward(self,pred,target, interpolate=True):
        valid_mask = (target > 0).detach()
        print(target.shape)
        if interpolate:
            pred = nn.functional.interpolate(pred, target.shape[-2:], mode='bilinear', align_corners=True)
        target = target[valid_mask]
        loss = self.ssim(target,pred)
        print("SSIM: ",loss)
        return loss
    
class Criterition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ssim = SSIMLoss()
        self.silog = SILogLoss()
    def forward(self,pred,target):
        # print("pass here")
        print("max pred: ",torch.max(pred))
        print("min pred: ",torch.min(pred))
        print("min target: ",torch.min(target))
        print("max target: ",torch.max(target))
        return self.ssim(pred,target)+self.silog(pred,target)