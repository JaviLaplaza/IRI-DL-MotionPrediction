import torch
import torch.nn as nn

class L_phase(torch.nn.Module):
    def __init__(self, wp=1):
        super(L_phase, self).__init__()
        self._wp = wp
        self.phase_loss = nn.CrossEntropyLoss()

    def forward(self, phase_pred, phase_target):
        return self._wi * self.phase_loss(phase_pred, phase_target.long())