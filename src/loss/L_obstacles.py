import torch

class L_obstacles(torch.nn.Module):
    def __init__(self, wo=1):
        super(L_obstacles, self).__init__()
        self._wo = wo

    def forward(self, xyz_estim, obstacles_):
        loss_obstacle = 0

        for i in range(3):
            intersection_left_hip = xyz_estim[:, :, [21, 22, 23]] - obstacles_[:, :, :, i]
            intersection_right_hip = xyz_estim[:, :, [24, 24, 26]] - obstacles_[:, :, :, i]

            if torch.any(torch.abs(intersection_left_hip[:, :, 0]) < 0.1):
                if torch.any(torch.abs(intersection_left_hip[:, :, 1]) < 0.1):
                    loss_obstacle = 50

            if torch.any(torch.abs(intersection_right_hip[:, :, 0]) < 0.1):
                if torch.any(torch.abs(intersection_right_hip[:, :, 1]) < 0.1):
                    loss_obstacle = 50

        return self._wo * loss_obstacle