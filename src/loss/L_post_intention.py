import torch
import torch.nn as nn

class L_post_intention(torch.nn.Module):
    def __init__(self, wi_post=1):
        super(L_post_intention, self).__init__()
        self._wi_post = wi_post
        self.intention_loss = nn.CrossEntropyLoss()

    def forward(self, intention_pred, intention_target):
        # print(f"post_intention_pred.shape: {intention_pred[0]}")
        # print(f"post_intention_target.shape: {intention_target[0]}")
        return self._wi_post * self.intention_loss(intention_pred, intention_target.long())