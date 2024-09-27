from torch.utils.data import Dataset
import numpy as np
#from h5py import File

import os
import scipy
import scipy.io as sio
from src.utils import data_utils, plots
from matplotlib import pyplot as plt
import torch

import itertools

from src.data.dataset import DatasetBase


class AssistiveRoboticsDataset(DatasetBase):
    def __init__(self, opt, is_for, subset, transform, dataset_type):
        super(AssistiveRoboticsDataset, self).__init__(opt, is_for, subset, transform, dataset_type)

        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self._name = 'assistive_robotics'

        self._init_meta(opt)


        seq_len = self.in_n + self.out_n
        #subs = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [10], [10]])

        # subs = np.array([[3, 4, 5, 6, 7, 8, 9], [10], [10]])
        scenarios = ["straight", "one_obstacle", "multiple_obstacles"]
        #human_scenarios = ["close", "delay", "free", "hold", "natural"]
        # acts = data_utils.define_actions(actions)

        # subs = np.array([[1], [11], [5]])
        # acts = ['walking']
        # 32 human3.6 joint name:
        joint_name = ["Head", "Chest",
                      "RightShoulder", "RightElbow", "RightHand",
                      "LeftShoulder", "LeftElbow", "LeftHand",
                      "Pelvis", "RightHip", "LeftHip"]
        """
        self._dimensions_to_use = [0, 1, 2, #nose (0, 1, 2)
                                   #4, 5, 6,       #left_eye_inner
                                   #8, 9, 10,      #left_eye
                                   #12, 13, 14,    #left_eye_outer
                                   #16, 17, 18,    #right_eye_inner
                                   #20, 21, 22,    #right_eye
                                   #24, 25, 26,    #right_eye_outer
                                   #28, 29, 30,    #left_ear
                                   #32, 33, 34,    #right_ear
                                   #36, 37, 38,    #mouth_left
                                   #40, 41, 42,    #mouth_right
                                   44, 45, 46,    #left_shoulder (3, 4, 5)
                                   48, 49, 50,    #right_shoulder (6, 7, 8)
                                   52, 53, 54,    #left_elbow (9, 10, 11)
                                   56, 57, 58,    #right_elbow (12, 13, 14)
                                   60, 61, 62,    #left_wrist (15, 16, 17)
                                   64, 65, 66,    #right_wrist (18, 19, 20)
                                   #68, 69, 70,    #left_pinky
                                   #72, 73, 74,    #right_pinky
                                   #76, 77, 78,    #left_index
                                   #80, 81, 82,    #right_index
                                   #84, 85, 86,    #left_thumb
                                   #88, 89, 90,    #right_thumb
                                   92, 93, 94,  #left_hip (21, 22, 23)
                                   96, 97, 98,  #right_hip (24, 25, 26)
                                   100, 101, 102, #left knee
                                   104, 105, 106, #right knee
                                   108, 109, 110, #left ankle
                                   112, 113, 114 #right ankle
                                   ]
        """

        self._dimensions_to_use = [0, 1, 2, #nose (0, 1, 2)
                                   #3, 4, 5,       #left_eye_inner
                                   #6, 7, 8,      #left_eye
                                   #9, 10, 11,    #left_eye_outer
                                   #12, 13, 14,    #right_eye_inner
                                   #15, 16, 17,    #right_eye
                                   #18, 19, 20,    #right_eye_outer
                                   #21, 22, 23,    #left_ear
                                   #24, 25, 26,    #right_ear
                                   #27, 28, 29,    #mouth_left
                                   #30, 31, 32,    #mouth_right
                                   33, 34, 35,    #left_shoulder (3, 4, 5)
                                   36, 37, 38,    #right_shoulder (6, 7, 8)
                                   39, 40, 41,    #left_elbow (9, 10, 11)
                                   42, 43, 44,    #right_elbow (12, 13, 14)
                                   45, 46, 47,    #left_wrist (15, 16, 17)
                                   48, 49, 50,    #right_wrist (18, 19, 20)
                                   #51, 52, 53,    #left_pinky
                                   #54, 55, 56,    #right_pinky
                                   #57, 58, 59,    #left_index
                                   #60, 61, 62,    #right_index
                                   #63, 64, 65,    #left_thumb
                                   #66, 67, 68,    #right_thumb
                                   69, 70, 71,  #left_hip (21, 22, 23)
                                   72, 73, 74,  #right_hip (24, 25, 26)
                                   # 75, 76, 77, #left knee
                                   # 78, 79, 80, #right knee
                                   # 81, 82, 83, #left ankle
                                   # 84, 85, 86 #right ankle
                                   # 87, 88, 89 #left heel
                                   # 90, 91, 92 #right heel
                                   # 93, 94, 95 #left foot index
                                   # 96, 97, 98 #right foot index
                                   # 99, 100, 101,  # chest
                                   # 102, 103, 104  # pelvis
                                   ]

        # self._end_effector_dims = [132, 133, 134]

        self._intention_dim = [132]

        # self._phase_dim = [135]


        # self._robot_path_dims = [138, 137]

        # subs = subs[self.split]
        key = 0

        for file in os.listdir(self.path_to_data):
                sub = int(file.split('_')[1])
                if sub in self.subs:
                    file_path = os.path.join(self.path_to_data, file)
                    try:
                        if file_path.endswith('.txt'):
                            print(f"Reading file {file_path}")
                            the_sequence = data_utils.readCSVasFloat(file_path)


                            # print(np.max(the_sequence), np.min(the_sequence))
                            # harvest_seq = the_sequence[np.squeeze(the_sequence[:, self._intention_dim] == 0)]
                            # harvest_left = harvest_seq[20:, [60, 61, 62]]
                            # harvest_right = harvest_seq[20:, [64, 65, 66]]
                            # harvest_middle = harvest_left + (harvest_right - harvest_left) / 2
                            # print(harvest_middle.shape)
                            # self._harvest_goal = np.mean(harvest_middle, axis=0)
                            # self._drop_goal = np.array([0, 0.6, 0.6])

                            # print(harvest_goal)
                            # if subj == 11:
                            #     plots.animate_mediapipe_full_body_sequence(the_sequence[:, self._dimensions_to_use], color='prediction', show=True)

                            # cond = the_sequence != np.ones((132,))*-1
                            # print(cond[])
                            # the_sequence = the_sequence[cond]
                            # print(the_sequence.shape)
                            # the_sequence = np.reshape(the_sequence, (-1, 132))
                            # print(the_sequence.shape)

                            # the_sequence = the_sequence[-75:, :]

                            # phase = True


                            #print(subj, scenario, file)
                            # if subj == 10:
                            #    if scenario == "multiple_obstacles": # arm, base, static
                            #         if file == "right_outer_free.txt":
                            #             print(the_sequence[:, self._dimensions_to_use].shape)
                            #             plots.animate_mediapipe_sequence(the_sequence[:, self._dimensions_to_use], show=True)


                            if len(the_sequence.shape) == 2:
                                n, d = the_sequence.shape
                                print(n, d)
                                even_list = range(0, n, self.sample_rate)
                                num_frames = len(even_list)
                                the_sequence = np.array(the_sequence[even_list, :])

                                the_sequence = the_sequence[:, self._dimensions_to_use]
                                the_sequence = torch.from_numpy(the_sequence).float()

                                # remove global rotation and translation
                                # the_sequence[:, 0:6] = 0
                                # p3d = data_utils.expmap2xyz_torch(the_sequence)

                                p3d = the_sequence

                                # self.p3d[(subj, action, subact)] = p3d.view(num_frames, -1).cpu().data.numpy()
                                self.p3d[key] = p3d.view(num_frames, -1).data.numpy()

                                valid_frames = np.arange(0, num_frames - seq_len + 1, self.skip_rate)

                                # tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                                tmp_data_idx_1 = [key] * len(valid_frames)
                                tmp_data_idx_2 = list(valid_frames)

                                tmp_data_idx_3 = file_path

                                # tmp_data_idx_4 = harvest_goal

                                # if scenario == 'one_obstacle':
                                #     tmp_data_idx_3 += 1

                                # elif scenario == 'multiple_obstacles':
                                #     tmp_data_idx_3 += 2

                                self.data_idx.extend(zip(tmp_data_idx_1,
                                                         tmp_data_idx_2,
                                                         itertools.repeat(tmp_data_idx_3)))

                                key += 1
                    except Exception as e:
                        print(e)



        # self.indices = [[] for _ in range(5)]
        # for i, x in enumerate(self.data_idx):
        #     key, start_frame, obstacle = x
        #     fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)

        #     intention = self.p3d[key][fs][:, self._intention_dim]
        #     intention_goal = torch.mode(torch.from_numpy(intention[-self.out_n:]), dim=0)[0]

        #     if intention_goal == 0: self.indices[0].append(i)
        #     if intention_goal == 1: self.indices[1].append(i)
        #     if intention_goal == 2: self.indices[2].append(i)
        #     if intention_goal == 3: self.indices[3].append(i)
        #    if intention_goal == 4: self.indices[4].append(i)


        # ignore constant joints and joints at same position with other joints
        #joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        #dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        #self.dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)

    def _init_meta(self, opt):
        self.path_to_data = opt[self._name]["data_dir"]

        self.in_n = opt[self._name]["input_n"]
        self.out_n = opt[self._name]["output_n"]

        self.sample_rate = 1
        self.skip_rate = opt[self._name]["skip_rate"]

        self.p3d = {}
        self.data_idx = []

        self.intention_condition = opt["networks"]["reg"]["hyper_params"]["intention_condition"]



        if self._is_for == "train":
            self.subs = self._opt[self._name]["train_ids"]
        elif self._is_for == "val":
            self.subs = self._opt[self._name]["val_ids"]
        elif self._is_for == "test":
            self.subs = self._opt[self._name]["test_ids"]
        else:
            raise ValueError(f"is_for={self._is_for} not valid")


    def intention_classes(self):
        return self.indices

    def get_stats(self):
        scenarios = ["straight", "one_obstacle", "multiple_obstacles"]

        lengths = []

        # subs = subs[self.split]
        key = 0
        for subj in self.subs:
            for scenario in scenarios:
                scenario_path = os.path.join(self.path_to_data, "S" + str(subj), scenario)
                for file in os.listdir(scenario_path):
                    if scenario == "straight":
                        file_path = os.path.join(scenario_path, file)

                        # try:
                        if file_path.endswith('.txt'):
                            the_sequence = data_utils.readCSVasFloat(file_path)

                            lengths.append(the_sequence.shape[0])




                        # except Exception as e:
                        #    print(e)

        print(f'avg video lenght: {np.mean(lengths)}')
        print(f'std video lenght: {np.std(lengths)}')

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame, file_path = self.data_idx[item]
        # print(f"harvest_goal: {harvest_goal}")

        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)

        xyz = self.p3d[key][fs] # [:, self._dimensions_to_use]

        xyz = np.reshape(xyz, (xyz.shape[0], -1, 3))
        xyz_ = np.copy(xyz)
        xyz[:, :, 1] = xyz_[:, :, 2]
        xyz[:, :, 2] = -xyz_[:, :, 1]
        xyz = np.reshape(xyz, (xyz.shape[0], -1))



        # harvest_goal = np.repeat(harvest_goal, len(xyz))


        # end_effector = self.p3d[key][fs][:, self._end_effector_dims]
        # phase = self.p3d[key][fs][:, self._phase_dim]
        if self.intention_condition:
            intention = self.p3d[key][fs][:, self._intention_dim]

            # goal = np.where(intention == 0, self._harvest_goal, self._drop_goal)
            # print(goal)

            sample = {'xyz': xyz,
                      'file_path': file_path,
                      'intention': intention
                      # 'goal': goal
                      }
        else:
            sample = {'xyz': xyz,
                      'file_path': file_path
                      # 'end_effector': end_effector,
                      #'obstacles': obstacle_position}
                      # 'obstacles': obstacles,
                      # 'phase': phase,
                      # 'phase_goal': phase_goal,
                      # 'intention': intention,
                      # 'intention_goal': intention_goal,
                      # 'robot_path': robot_path
                    }

        if self._transform is not None:
            sample = self._transform(sample)

        return sample

        #return self.p3d[key][fs], obstacle_position


if __name__ == "__main__":
    from src.options.config_parser import ConfigParser
    from src.utils.data_utils import iri_discretize_pose,  iri_undiscretize_pose
    import torch
    from src.utils.plots import plot_3d_points

    opt = ConfigParser().get_config()
    transforms = opt["transforms"]["rand_mediapipe_skel_rotate"]
    dataset = AssistiveRoboticsDataset(opt, is_for='train', subset=0, transform=None, dataset_type=0)


    sample = dataset[0]
    #sample = dataset[250]

    print(f"sample['xyz']: {sample['xyz']}")

    xyz = sample['xyz'].reshape((sample['xyz'].shape[0], -1, 3))

    labels = np.linspace(0, xyz.shape[1]-1, xyz.shape[1])
    print(labels)

    for frame in xyz:
        x_t = frame[:, 0]
        y_t = frame[:, 1]
        z_t = frame[:, 2]
        plot_3d_points(x_t, y_t, z_t, labels)
    # print(f"sample.shape: {sample.shape}")

    # print(f"sample file path: {sample['file_path']}")

    # print((f"sample['intention']: {sample['intention']}"))

    n0 = n1 = n2 = n3 = 0

    """
    for sample in dataset:
        if sample['intention_goal'] == 0:
            n0 += 1

        elif sample['intention_goal'] == 1:
            n1 += 1

        elif sample['intention_goal'] == 2:
            n2 += 1

        elif sample['intention_goal'] == 3:
            n3 += 1
    """


    # print(dataset.intention_classes())
    # print(n0, n1, n2, n3)

    # dataset.get_stats()


    upper_body = sample['xyz'] #[:, dataset._dimensions_to_use]
    # end_effector = sample['end_effector']
    # obstacles = sample['obstacles']
    #plots.animate_iri_handover_sequence(sample[:, dataset._dimensions_to_use], show=True)

    upper_body = np.expand_dims(upper_body, axis=0)
    #first_frame = upper_body[:, 0]
    # print(f"upper_body.shape: {upper_body.shape}")
    # print(f"end_effector.shape: {end_effector.shape}")
    # print(f"obstacles.shape: {obstacles.shape}")

    upper_body = torch.from_numpy(upper_body)
    # print(upper_body.shape)
    # print(obstacles.shape)
    #print(obstacles)
    #upper_body, first_frame = iri_discretize_pose(upper_body)

    #upper_body, _ = iri_undiscretize_pose(upper_body.cpu(), first_frame, n_bins=100)

    plots.animate_mediapipe_full_body_sequence(upper_body[0], color='prediction', show=True)


