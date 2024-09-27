import torch

class L2(torch.nn.Module):
    def __init__(self, wx=1, wy=1, wz=1):
        super(L2, self).__init__()
        self._wx = wx
        self._wy = wy
        self._wz = wz

    def forward(self, xyz_estim, xyz_target):
        num_features = xyz_estim.shape[-1]
        """
        loss_x = torch.mean(torch.sum(torch.abs(xyz_estim - xyz_target)[:, :, list(range(0, num_features, 3))], dim=2))
        loss_y = torch.mean(torch.sum(torch.abs(xyz_estim - xyz_target)[:, :, list(range(1, num_features, 3))], dim=2))
        loss_z = torch.mean(torch.sum(torch.abs(xyz_estim - xyz_target)[:, :, list(range(2, num_features, 3))], dim=2))
        """
        loss_x = torch.mean(torch.norm(xyz_estim[:, :, list(range(0, num_features, 3))] - xyz_target[:, :, list(range(0, num_features, 3))], dim=2))
        loss_y = torch.mean(torch.norm(xyz_estim[:, :, list(range(1, num_features, 3))] - xyz_target[:, :, list(range(1, num_features, 3))], dim=2))
        loss_z = torch.mean(torch.norm(xyz_estim[:, :, list(range(2, num_features, 3))] - xyz_target[:, :, list(range(2, num_features, 3))], dim=2))

        # loss_xyz = torch.mean(torch.sum(torch.abs(xyz_estim - xyz_target), dim=2))

        loss_xyz = self._wx * loss_x + self._wy * loss_y + self._wz * loss_z

        # loss_xyz = self._wx * loss_x

        return loss_xyz
