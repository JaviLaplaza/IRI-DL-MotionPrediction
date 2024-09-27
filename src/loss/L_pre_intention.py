import torch
import torch.nn as nn

class L_pre_intention(torch.nn.Module):
    def __init__(self, wi_pre=1):
        super(L_pre_intention, self).__init__()
        self._wi_pre = wi_pre
        self.intention_loss = nn.CrossEntropyLoss()

    def forward(self, intention_pred, intention_target):
        # print(f"pre_intention_pred.shape: {intention_pred[0]}")
        # print(f"pre_intention_target.shape: {intention_target[0]}")
        return self._wi_pre * self.intention_loss(intention_pred, intention_target.long())