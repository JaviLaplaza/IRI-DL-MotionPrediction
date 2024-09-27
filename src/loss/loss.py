import torch.nn as nn

class LossFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(loss_name, *args, **kwargs):
        if loss_name == 'L2':
            from .L2 import L2
            loss = L2(*args, **kwargs)
        elif loss_name == 'L_REE':
            from .L_REE import L_REE
            loss = L_REE(*args, **kwargs)
        elif loss_name == 'L_free_hand':
            from .L_free_hand import L_free_hand
            loss = L_free_hand(*args, **kwargs)
        elif loss_name == 'L_obstacles':
            from .L_obstacles import L_obstacles
            loss = L_obstacles(*args, **kwargs)
        elif loss_name == 'L_phase':
            from .L_phase import L_phase
            loss = L_phase(*args, **kwargs)
        elif loss_name == 'L_intention':
            from .L_intention import L_intention
            loss = L_intention(*args, **kwargs)
        elif loss_name == 'L_temporal':
            from .L_temporal import L_temporal
            loss = L_temporal(*args, **kwargs)
        elif loss_name == 'L_last_pose':
            from .L_last_pose import L_last
            loss = L_last(*args, **kwargs)
        elif loss_name == 'L_total':
            from .L_total import L_total
            loss = L_total(*args, **kwargs)
        else:
            raise ValueError(f"Loss %s not recognized." % loss_name)

        return loss


class LossBase(nn.Module):
    def __init__(self):
        super(LossBase, self).__init__()
        self._name = 'BaseLoss'

    @property
    def name(self):
        return self._name