import torch
from torch.nn.functional import one_hot
from torcheval.metrics.functional import multiclass_f1_score
from torcheval.metrics import MulticlassConfusionMatrix

from collections import OrderedDict
from src.utils import util
from .models import BaseModel
from src.networks.networks import NetworksFactory
from src.loss.loss import LossFactory
from src.utils.plots import animate_mediapipe_target_and_prediction, animate_mediapipe_sequence, \
    animate_h36m_sequence, animate_h36m_target_and_prediction, animate_mediapipe_full_body_sequence, animate_mediapipe_full_body_sequence_pred_and_target

import numpy as np

class HistRepItself(BaseModel):
    def __init__(self, opt):
        super(HistRepItself, self).__init__(opt)
        self._name = 'HistRepItself'

        # init input params
        self._init_set_input_params()

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt["model"]["load_epoch"] > 0:
            self.load()

        if self._opt["canopies"]["pretrained"] > 0:
            # self.load_pretrained()
            epoch = self._opt["canopies"]["pretrained"]
            pretrained_model = torch.load(f"experiments/HistRepItself/h36m_2/checkpoints/net_epoch_{epoch}_id_best_nn_reg.pth", map_location=lambda storage, loc: storage)
            self._reg.load_state_dict(pretrained_model)
            print("Pretrained model loaded")

        # init losses
        if self._is_train:
            self._init_losses()

        # prefetch inputs
        self._init_prefetch_inputs()

    def _init_set_input_params(self):
        self._B = self._opt[self._dataset_type]["batch_size"]               # batch
        self._features = self._opt[self._dataset_name]["features"]               # number of joints (9) x 3 dimensions
        self._in_n = self._opt[self._dataset_name]["input_n"]
        self._out_n = self._opt[self._dataset_name]["output_n"]
        self._seq_n = self._in_n + self._out_n # seq length
        self._kernel_size = self._opt["networks"]["reg"]["hyper_params"]["kernel_size"]
        self._itera = self._opt["networks"]["reg"]["hyper_params"]["itera"]
        # if self._opt[self._dataset_type] in ['mediapipe_handover', 'canopies']:
        self._goal_condition = self._opt["networks"]["reg"]["hyper_params"]["goal_condition"]
        self._obstacle_condition = self._opt["networks"]["reg"]["hyper_params"]["obstacle_condition"]
        self._robot_path_condition = self._opt["networks"]["reg"]["hyper_params"]["robot_path_condition"]
        self._phase_condition = self._opt["networks"]["reg"]["hyper_params"]["phase_condition"]
        self._intention_condition = self._opt["networks"]["reg"]["hyper_params"]["intention_condition"]
        self._phase_prediction = self._opt["networks"]["reg"]["hyper_params"]["phase_prediction"]
        self._intention_prediction = self._opt["networks"]["reg"]["hyper_params"]["intention_prediction"]

        # self._Ct = self._opt[self._dataset_type]["target_nc"] * self._B     # num channels target

    def _init_create_networks(self):
        # create reg
        reg_type = self._opt["networks"]["reg"]["type"]
        reg_hyper_params = self._opt["networks"]["reg"]["hyper_params"]
        self._reg = NetworksFactory.get_by_name(reg_type, **reg_hyper_params, device=self._device_master) #.to(self._device_master)
        self._reg = torch.nn.DataParallel(self._reg, device_ids=self._reg_gpus_ids)

    def _init_train_vars(self):
        self._current_lr = self._opt["train"]["reg_lr"]
        # self._optimizer = torch.optim.SGD(self._reg.parameters(), lr=self._current_lr)
        self._optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self._reg.parameters()), lr=self._current_lr)

    def _init_losses(self):
        loss_type = self._opt["losses"]["type"]
        weights = self._opt["losses"]["weights"]
        self._criterion = LossFactory.get_by_name(loss_type, **weights).to(self._device_master)
        # self._criterion = torch.nn.CrossEntropyLoss().to(self._device_master)

    def _init_prefetch_inputs(self):
        self._input_seq = torch.zeros([self._B, self._seq_n, self._features]).to(self._device_master)
        self._goal = torch.zeros([self._B, self._seq_n, 3]).to(self._device_master)
        self._input_obstacles = torch.zeros([self._B, self._seq_n, 3 * 3]).to(self._device_master)
        self._input_obstacles_ = torch.zeros([self._B, self._seq_n, 3, 3]).to(self._device_master)
        self._input_robot_path = torch.zeros([self._B, self._seq_n, 2]).to(self._device_master)
        self._input_phase = torch.zeros([self._B, self._out_n, 1]).to(self._device_master)
        self._input_phase_goal = torch.zeros([self._B, 1]).to(self._device_master)
        self._input_intention = torch.zeros([self._B, self._out_n, 1]).to(self._device_master)
        self._input_intention_goal = torch.zeros([self._B, 1]).to(self._device_master)

        self._input_target = torch.zeros([self._B, self._out_n, self._features]).to(self._device_master)

    def set_input(self, input):
        # copy values
        self._input_seq.copy_(input['xyz'])

        if self._goal_condition:
            self._goal.copy_(input['goal'])
        if self._obstacle_condition:
            self._input_obstacles.copy_(input['obstacles'].view(-1, self._seq_n, 9).float())
            self._input_obstacles_.copy_(input['obstacles'])
            self._input_obstacles = self._input_obstacles
        if self._robot_path_condition:
            self._input_robot_path.copy_(input['robot_path'])
        if self._phase_condition:
            self._input_phase.copy_(input['phase'][:, -self._out_n:])
        if self._intention_condition:
            self._input_intention_goal.copy_(input['intention_goal'])  # [:, -self._out_n:])

        if self._phase_prediction:
            self._input_phase_goal.copy_(input['phase_goal'][:, -self._out_n:])
        if self._intention_prediction:
            self._input_intention.copy_(input['intention'][:, -self._out_n:])

            # self._input_intention_goal.copy_(input['intention_goal'])
            # one_hot_format = False
            # if one_hot_format:
            #     intention_goal = torch.squeeze(one_hot(input['intention_goal'].to(torch.int64), num_classes=5).float(), dim=1)
            #     self._input_intention_goal.copy_(intention_goal)

        self._input_target.copy_(input['xyz'][:, -self._out_n:])

        # move to gpu
        self._input_seq = self._input_seq.to(self._device_master)
        self._goal = self._goal.to(self._device_master)
        self._input_obstacles = self._input_obstacles.to(self._device_master)
        self._input_obstacles_ = self._input_obstacles_.to(self._device_master)
        self._input_phase = self._input_phase.to(self._device_master)
        self._input_intention = self._input_intention.to(self._device_master)

        # self._input_img = self._input_img.to(self._device_master)
        self._input_target = self._input_target.to(self._device_master)

    def set_train(self):
        self._reg.train()
        self._is_train = True

    def set_eval(self):
        self._reg.eval()
        self._is_train = False

    def evaluate(self):
        # set model to eval
        is_train = self._is_train
        if is_train:
            self.set_eval()

        # estimate object categories
        with torch.no_grad():
            self.forward(keep_data_for_visuals=True, estimate_loss=False)
            # eval = np.transpose(self._vis_input_img, (1, 2, 0))
            eval = self._vis_input_img[0]

        # set model back to train if necessary
        if is_train:
            self.set_train()

        return eval

    def optimize_parameters(self, keep_data_for_visuals=False):
        if self._is_train:

            # calculate loss
            loss = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            # optimize
            self._optimizer.zero_grad()
            loss.backward()
            # grad_norm = nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            self._optimizer.step()

        else:
            raise ValueError('Trying to optimize in non-training mode!')

    def forward(self, keep_data_for_visuals=False, estimate_loss=True):
        torch.autograd.set_detect_anomaly(True)
        # generate img
        "xyz_estim_all, pre_intention_estim, phase_estim, intention_estim = \
            self._estimate(self._input_seq, input_n=self._in_n, output_n=self._out_n, itera=self._itera, \
                            goal=self._goal, obstacles=self._input_obstacles, robot_path=self._input_robot_path, \
                            phase=self._input_phase, intention=self._input_intention, phase_goal=self._input_phase_goal, \
                            intention_goal=self._input_intention_goal)"
        xyz_estim_all, pre_intention_estim, phase_estim, intention_estim = \
            self._estimate(self._input_seq, input_n=self._in_n, output_n=self._out_n, itera=self._itera, \
                            goal = self._goal, obstacles = self._input_obstacles, robot_path = self._input_robot_path, \
                            phase = self._input_phase, intention = self._input_phase_goal, \
                            phase_goal = self._input_phase_goal, intention_goal = self._input_intention_goal)

        xyz_estim = xyz_estim_all[:, :, 0]

        xyz_estim = xyz_estim[:, self._kernel_size:]
        # phase_estim = phase_estim[:, :, self._in_n:].permute((0, 2, 1))



        # print(f"np.max(self._input_target): {torch.max(self._input_target)}, np.min(self._input_target): {torch.min(self._input_target)}")
        # print(f"np.max(xyz_estim): {torch.max(xyz_estim)}, np.min(xyz_estim): {torch.min(xyz_estim)}")

        # if self._intention_prediction:
        # print(intention_estim.shape)
        # intention_estim = intention_estim[:, -self._out_n:]
        # print(intention_estim.shape)
        # print(f'intention_estim.shape: {intention_estim.shape}')
        # print(f'torch.max(intention_estim): {torch.argmax(intention_estim, dim=1)}')

        # print(intention_estim.shape)
        # print(self._input_intention.shape)


        # estimate loss
        if estimate_loss:
            # print(xyz_estim[0])
            # print(self._input_target[0])

            self._loss_gt = self._criterion(xyz_estim, self._input_target,
                                            obstacles=self._input_obstacles_[:, -self._out_n:],
                                            phase_estim=phase_estim, phase_target=self._input_phase,
                                            intention_estim=intention_estim,
                                            # intention_target=self._input_intention.squeeze(2),
                                            intention_target=self._input_intention_goal.squeeze(1),
                                            intention_pred=pre_intention_estim, intention_goal=self._input_intention_goal.squeeze(1))

            # self._loss_gt = self._criterion(xyz_estim, self._input_target)
            # print(f"self._loss_gt: {self._loss_gt}")

            total_loss = self._loss_gt

        else:
            total_loss = -1

        # print(f"FB Error sequence: {torch.norm(xyz_estim - self._input_target, dim=1)[0]}")
        self.current_body_metric = torch.mean(
                                    torch.mean(torch.norm(
                                                xyz_estim[:, :, list(range(0, xyz_estim.shape[-1] - 12, 3))] -
                                                self._input_target[:, :, list(range(0, xyz_estim.shape[-1] - 12, 3))],
                                              dim=2),
                                    dim=1)) # / xyz_estim.shape[0]
        # print(self.current_body_metric)

        show_per_windows_results = True
        if show_per_windows_results:
            xyz_diff = torch.mean(torch.mean(xyz_estim[:, :, list(range(0, xyz_estim.shape[-1] - 12, 3))] - self._input_target[:, :, list(range(0, xyz_estim.shape[-1] - 12, 3))], dim=2), dim=0)
            # print(xyz_diff)

        # print(f"RH Error sequence: {torch.norm((xyz_estim - self._input_target)[:, :, [18, 19, 20]], dim=2)[0]}")
        self.current_right_hand_metric = torch.mean(torch.norm(xyz_estim[:, :, [18, 19, 20]] - self._input_target[:, :, [18, 19, 20]], dim=2)) # / xyz_estim.shape[0]

        """
        if True:
            translation_vec = xyz_estim[:, :, [21, 22, 23]] - self._input_target[:, :, [21, 22, 23]]
            xyz_estim_translated = xyz_estim - translation_vec.repeat(1, 1, 13)

            print(torch.mean(torch.norm(xyz_estim_translated - self._input_target, dim=1)))

        """

        if self._intention_condition:
            # print(intention_pred_)
            self.current_preintention_acc = multiclass_f1_score(input=torch.reshape(pre_intention_estim, (-1, 4)),
                                                                target=self._input_intention_goal.squeeze(1),
                                                                num_classes=4)

            # print(f'current pre intention acc: {self.current_preintention_acc}')

        if self._intention_prediction:
            # intention_pred_ = torch.argmax(intention_estim, dim=1)
            # current_intention_acc = (intention_pred_ == self._input_intention.squeeze(2)).sum() / torch.numel(intention_pred_)
            # current_intention_acc = current_intention_acc.item()
            # print(f'current intention acc: {current_intention_acc}')


            self.current_postintention_acc = multiclass_f1_score(input=torch.reshape(intention_estim, (-1, 4)),
                                                                # target=torch.flatten(self._input_intention.squeeze(2)),
                                                                target=self._input_intention_goal.squeeze(1),
                                                                num_classes=4)

            # print(f'current post intention acc: {self.current_postintention_acc}')

        # keep visuals
        if keep_data_for_visuals:
            self._keep_data(xyz_estim, pre_intention_estim, intention_estim)

        return total_loss

    def _estimate(self, input, **kwargs):
        return self._reg.forward(input, **kwargs)

    def _keep_data(self, estim, pre_intention_estim=0, intention_estim=0):
        index = torch.randint(0, estim.shape[0], (1,))
        target = self._input_target[index].detach().cpu().numpy()[0]
        predicted = estim[index].detach().cpu().numpy()[0]
        pre_intention_estim_idx = []
        if self._intention_condition:
            pre_intention_estim_idx = torch.argmax(pre_intention_estim[index])

        intention_estim_idx = []
        if self._intention_prediction:
            intention_estim_idx = torch.argmax(intention_estim[index])

        # vis_img = util.tensor2im(self._input_img.detach(), unnormalize=True, to_numpy=True)
        video_buf, fig, ax, frame = animate_mediapipe_target_and_prediction(target, predicted, show=False,
                                                                            pre_intention_estim=pre_intention_estim_idx,
                                                                            intention_estim=intention_estim_idx)
        # video_buf, fig, ax, frame = animate_h36m_sequence(predicted)
        # video_buf, fig, ax, frame = animate_h36m_target_and_prediction(target, predicted)
        # video_buf, fig, ax, frame = animate_mediapipe_full_body_sequence_pred_and_target(predicted, target)

        video_buf = torch.stack(video_buf, dim=1)
        self._vis_input_img = video_buf

    def get_inputs(self):
        return self._input_seq, self._goal, self._input_obstacles, self._input_phase, self._input_intention, self._input_target

    def get_image_paths(self):
        return OrderedDict()

    def get_current_errors(self):
        loss_dict = OrderedDict()
        loss_dict["loss_gt"] = self._loss_gt.item()
        return loss_dict

    def get_current_body_metrics(self):
        return OrderedDict([('body_metrics', self.current_body_metric.cpu())])

    def get_current_right_hand_metrics(self):
        return OrderedDict([('right_hand_metrics', self.current_right_hand_metric.cpu())])

    def get_current_preintention_metrics(self):
        return OrderedDict([('preintention_metrics', self.current_preintention_acc.cpu())])

    def get_current_postintention_metrics(self):
        return OrderedDict([('postintention_metrics', self.current_postintention_acc.cpu())])

    def get_current_scalars(self):
        return OrderedDict([('lr', self._current_lr)])

    def get_current_visuals(self):
        visuals = OrderedDict()
        visuals["1_estim_img"] = self._vis_input_img
        return visuals

    def save(self, epoch_label, save_type, do_remove_prev=True):
        # save networks
        self._save_network(self._reg, 'nn_reg', epoch_label, save_type, do_remove_prev)
        self._save_optimizer(self._optimizer, 'o_reg', epoch_label, save_type, do_remove_prev)

    def save_best(self, epoch_label, save_type, do_remove_prev=True):
        # save networks
        self._save_network(self._reg, 'best_nn_reg', epoch_label, save_type, do_remove_prev)
        self._save_optimizer(self._optimizer, 'best_o_reg', epoch_label, save_type, do_remove_prev)


    def load(self):
        # load networks
        load_epoch = self._opt["model"]["load_epoch"]
        self._load_network(self._reg, 'nn_reg', load_epoch)
        if self._is_train:
            self._load_optimizer(self._optimizer, "o_reg", load_epoch)

    def load_pretrained(self):
        # load networks
        load_epoch = self._opt["canopies"]["pretrained"]
        self._load_network(self._reg, 'best_nn_reg', load_epoch)


    def update_learning_rate(self, curr_epoch):
        initial_lr = float(self._opt["train"]["reg_lr"])
        nepochs_no_decay = self._opt["train"]["nepochs_no_decay"]
        nepochs_decay = self._opt["train"]["nepochs_decay"]

        # update lr
        if curr_epoch <= nepochs_no_decay:
            self._current_lr = initial_lr
        else:
            new_lr = self._lr_linear(self._current_lr, nepochs_decay, initial_lr)
            self._update_learning_rate(self._optimizer, "reg", self._current_lr, new_lr)
            self._current_lr = new_lr
