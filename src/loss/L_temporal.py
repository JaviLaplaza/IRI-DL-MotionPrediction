import torch

class L_temporal(torch.nn.Module):
    def __init__(self, wt=1):
        super(L_temporal, self).__init__()
        self._wt = wt

    def forward(self, xyz_estim):
        temporal_delta = torch.diff(xyz_estim, dim=1)
        loss_temporal = torch.mean(torch.sum(torch.abs(temporal_delta), dim=2))

        return self._wt * loss_temporal
