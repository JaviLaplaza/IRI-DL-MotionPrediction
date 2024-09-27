import torch
from src.loss.L2 import L2

class L_total(torch.nn.Module):
    def __init__(self, wx=1, wy=1, wz=1, we=-1, wo=-1, wf=-1, wp=-1, wi_pre=-1, wi_post=-1, wt=-1, wl=-1):
        super(L_total, self).__init__()
        self._wx = wx
        self._wy = wy
        self._wz = wz
        self._we = we
        self._wo = wo
        self._wf = wf
        self._wp = wp
        self._wi_post = wi_post
        self._wi_pre = wi_pre
        self._wt = wt
        self._wl = wl

        self.loss_xyz = L2(wx=self._wx, wy=self._wy, wz=self._wz)
        self.intention_loss = torch.nn.CrossEntropyLoss()

        if self._we > 0:
            from src.loss.L_REE import L_REE
            self.loss_ree = L_REE(we=we)

        if self._wo > 0:
            from src.loss.L_obstacles import L_obstacles
            self.loss_obstacles = L_obstacles(wo=wo)

        if self._wf > 0:
            from src.loss.L_free_hand import L_free_hand
            self.loss_free_hand = L_free_hand(wf=wf)

        if self._wp > 0:
            from src.loss.L_phase import L_phase
            self.loss_phase = L_phase(wp=wp)

        if self._wi_post > 0:
            from src.loss.L_post_intention import L_post_intention
            self.loss_post_intention = L_post_intention(wi_post=wi_post)

        if self._wi_pre > 0:
            from src.loss.L_pre_intention import L_pre_intention
            self.loss_pre_intention = L_pre_intention(wi_pre=wi_pre)

        if self._wt > 0:
            from src.loss.L_temporal import L_temporal
            self.loss_temporal = L_temporal(wt=wt)

        if self._wl > 0:
            from src.loss.L_last_pose import L_last
            self.loss_last = L_last(wl=wl)


    def forward(self, xyz_estim, xyz_target, obstacles=torch.Tensor([]), phase_estim=torch.Tensor([]),
                phase_target=torch.Tensor([]), intention_estim=torch.Tensor([]), intention_target=torch.Tensor([]),
                intention_pred=torch.Tensor([]), intention_goal=torch.Tensor([])):
        loss_total = self.loss_xyz(xyz_estim, xyz_target)
        # print(f"loss xyz: {self.loss_xyz(xyz_estim, xyz_target)}")

        # intention_loss = self.intention_loss(intention_estim, intention_target.long())
        intention_loss = 0
        # print(f"loss intention: {self.intention_loss(intention_estim, intention_target.long())}")
        loss_total += intention_loss

        if self._we > 0:
            loss_total += self.loss_ree(xyz_estim, xyz_target)
            # print(f"loss ree: {self.loss_ree(xyz_estim, xyz_target)}")

        if self._wo > 0:
            loss_total += self.loss_obstacles(xyz_estim, obstacles)
            # print(f"loss obstacles: {self.loss_obstacles(xyz_estim, obstacles)}")

        if self._wf > 0:
            loss_total += self.loss_free_hand(xyz_estim, xyz_target)
            # print(f"loss free hand: {self.loss_free_hand(xyz_estim, xyz_target)}")

        if self._wp > 0:
            loss_total += self.loss_phase(phase_estim, phase_target)
            # print(f"loss phase: {self.loss_phase(phase_estim, phase_target)}")

        if self._wi_post > 0:
            loss_total += self.loss_post_intention(intention_estim, intention_target)
            # print(f"loss intention post: {self.loss_intention(intention_estim, intention_target)}")

        if self._wi_pre > 0:
            loss_total += self.loss_pre_intention(intention_pred, intention_goal)
            # print(f"loss intention pre: {self.loss_pre_intention(intention_pred, intention_goal)}")

        if self._wt > 0:
            loss_total += self.loss_temporal(xyz_estim)
            # print(f"loss temporal: {self.loss_temporal(xyz_estim)}")

        if self._wl > 0:
            loss_total += self.loss_last(xyz_estim, xyz_target)
            # print(f"loss last: {self.loss_last(xyz_estim, xyz_target)}")

        return loss_total
