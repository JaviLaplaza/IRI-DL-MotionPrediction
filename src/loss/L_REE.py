import torch

class L_REE(torch.nn.Module):
    def __init__(self, we=1):
        super(L_REE, self).__init__()
        self._we = we

    def forward(self, xyz_estim, xyz_target):
        loss_end_effector = torch.mean(torch.sum(torch.abs(xyz_estim - xyz_target)[:, :, [18, 19, 20]], dim=2))

        return self._we * loss_end_effector