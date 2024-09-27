import torch

class L_last(torch.nn.Module):
    def __init__(self, wl=1):
        super(L_last, self).__init__()
        self._wl = wl

    def forward(self, xyz_estim, xyz_target):
        xyz_estim_last = xyz_estim[:, -1]
        xyz_target_last = xyz_target[:, -1]
        loss_last = torch.mean(torch.norm(xyz_estim_last - xyz_target_last, dim=1))

        loss_last = self._wl * loss_last

        return loss_last
