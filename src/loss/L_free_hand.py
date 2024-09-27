import torch

class L_free_hand(torch.nn.Module):
    def __init__(self, wf=1):
        super(L_free_hand, self).__init__()
        self._wf = wf

    def forward(self, xyz_estim, xyz_target):
        loss_free_hand = torch.mean(torch.sum(torch.abs(xyz_estim - xyz_target)[:, :, [15, 16, 17]], dim=2))

        return self._wf * loss_free_hand