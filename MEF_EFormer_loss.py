import torch
import torch.nn as nn
from kornia.losses import SSIMLoss
import lpips

class MEF_EFormer_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='mean')
        self.l2 = nn.MSELoss(reduction='mean')
        self.ssim = SSIMLoss(window_size=11,max_val=1.0,reduction = 'mean')

        self.lpips_loss = lpips.LPIPS(net='vgg') 

        self.weights = {
            "l1": 1.0,
            "l2": 1.0,
            "ssim": 0.6,
            "lpips": 1.5,
            }
        

    def forward(self, fused: torch.Tensor, target: torch.Tensor):
        loss_l1 = self.l1(fused, target)
        loss_l2 = self.l2(fused, target)
  
        fused_bound = fused.clamp(min=0, max=1)
        target_bound = target.clamp(min=0, max=1)
        loss_ssim = self.ssim(fused_bound, target_bound)
        fused_lpips = fused_bound * 2 - 1  # [0,1] -> [-1,1]
        target_lpips = target_bound * 2 - 1  # [0,1] -> [-1,1]
        loss_lpips = self.lpips_loss(fused_lpips, target_lpips).mean()


        total_loss = (
            self.weights["l1"] * loss_l1 +
            self.weights["l2"] * loss_l2 +
            self.weights["ssim"] * loss_ssim +
            self.weights["lpips"] * loss_lpips
        )

        return total_loss