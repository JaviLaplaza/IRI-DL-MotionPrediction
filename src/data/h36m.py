from torch.utils.data import Dataset
from src.data.dataset import DatasetBase

import copy

import numpy as np

#from h5py import File

import os
import scipy
import scipy.io as sio
from src.utils import forward_kinematics, data_utils, plots
from matplotlib import pyplot as plt
import torch

class H36M(DatasetBase):
  """
  Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270
  Args
    path_to_dataset: string. directory where the data resides
    subjects: list of numbers. The subjects to load
    actions: list of string. The actions to load
    one_hot: Whether to add a one-hot encoding to the data
  Returns
    trainData: dictionary with k:v
      k=(subject, action, subaction, 'even'), v=(nxd) un-normalized data
    completeData: nxd matrix with all the data. Used to normlization stats
  """

  def __init__(self, opt, is_for, subset, transform, dataset_type=0):
    super(H36M, self).__init__(opt, is_for, subset, transform, dataset_type=0)
    """
    :param path_to_dataset:
    :param actions:
    :param input_n:
    :param output_n:
    :param dct_used:
    :param split: 0 train, 1 testing, 2 validation
    :param sample_rate:
    """
    self._name = 'h36m'
    self._is_for = is_for

    self._subset = subset
    # self._actions = actions

    self._init_meta(opt)

    seq_len = self.in_n + self.out_n

    """
    self._dims_to_use = [0, 1, 2,     # translation pelvis
                         3, 4, 5,     # rotation pelvis
                         6, 7, 8,     # right hip
                         9, 10, 11,   # right knee
                         12, 13, 14,  # right toe
                         15, 16, 17,  # right foot
                         18, 19, 20,  # right foot
                         21, 22, 23,  # left hip
                         24, 25, 26,  # left knee
                         27, 28, 29,  # left toe
                         30, 31, 32,  # left foot
                         33, 34, 35,  # left foot
                         36, 37, 38,  # pelvis
                         39, 40, 41,  # belly
                         42, 43, 44,  # neck
                         45, 46, 47,  # head
                         48, 49, 50,  # head
                         51, 52, 53,  # head
                         54, 55, 56,  # left shoulder
                         57, 58, 59,  # left elbow
                         60, 61, 62,  # left hand
                         63, 64, 65,  # left hand
                         66, 67, 68,  # left hand
                         69, 70, 71,  # left hand
                         72, 73, 74,  # left hand
                         75, 76, 77,  # left hand
                         78, 79, 80,  # right shoulder
                         81, 82, 83,  # right elbow
                         84, 85, 86,  # right hand
                         87, 88, 89,  # right hand
                         90, 91, 92,  # right hand
                         93, 94, 95,  # right hand
                         96, 97, 98   # right hand
                         ]

    """

    self._dims_to_use = [14, 17, 25, 18, 26, 19, 27, 6, 1, 7, 2, 8, 3]
    # self._dims_to_use = [14, 17, 25, 18, 26, 19, 27, 6, 1]

    nactions = len(self._actions)

    trainData = {}
    completeData = []
    key = 0
    for subj in self._subjects:
      for action_idx in np.arange(len(self._actions)):

        action = self._actions[action_idx]

        for subact in [1, 2]:  # subactions

          print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

          filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_dataset, subj, action, subact)
          action_sequence = self.readCSVasFloat(filename)

          n, d = action_sequence.shape
          even_list = range(0, n, self.sample_rate)
          num_frames = len(even_list)

          if self.one_hot:
            # Add a one-hot encoding at the end of the representation
            the_sequence = np.zeros((len(even_list), d + nactions), dtype=float)
            the_sequence[:, 0:d] = action_sequence[even_list, :]
            the_sequence[:, d + action_idx] = 1
            trainData[(subj, action, subact, 'even')] = the_sequence

          else:
            action_sequence = action_sequence[even_list, :]
            action_sequence[:, [0, 1, 2]] = action_sequence[:, [0, 1, 2]] # * 100
            action_sequence = torch.from_numpy(action_sequence).float().cpu()
            print(action_sequence.shape)
            # p3d = data_utils.expmap2xyz_torch(action_sequence)[:, self._dims_to_use]
            # p3d = data_utils.expmap2xyz(action_sequence)[:, self._dims_to_use]

            # print(action_sequence[:, :3])

            p3d = []
            for pose in action_sequence:
              if p3d == []:
                p3d = data_utils.expmap2xyz(pose) # [:, self._dims_to_use]
                p3d = np.reshape(p3d, (1, 96))

              else:
                pose_xyz = data_utils.expmap2xyz(pose) # [:, self._dims_to_use]
                pose_xyz = np.reshape(pose_xyz, (1, 96))
                p3d = np.concatenate((p3d, pose_xyz), axis=0)
              # print(i, p3d.shape)

            p3d = np.reshape(p3d, (p3d.shape[0], -1, 3))
            p3d = p3d[:, self._dims_to_use]
            # p3d_ = torch.clone(p3d)
            # p3d[:, :, 1] = p3d_[:, :, 0]
            # p3d[:, :, 2] = p3d_[:, :, 1] + 800
            # p3d[:, :, 0] = p3d_[:, :, 2]
            # p3d[:, :, 1] = p3d_[:, :, 1] + 800

            camera_frame = np.array([0, -250, -2000])

            # p3d += camera_frame
            p3d[:, :, 2] = -p3d[:, :, 2]

            # p3d = self.change_reference(p3d, camera_frame)

            p3d = np.reshape(p3d, (num_frames, -1))

            p3d /= 1000
            self.p3d[key] = torch.from_numpy(p3d).view(num_frames, -1).data.numpy()
            # self.p3d[key] = p3d.view(num_frames, -1).data.numpy()

            plots.animate_mediapipe_full_body_sequence(p3d, color='prediction',
                                                       show=True)

            valid_frames = np.arange(0, num_frames - seq_len + 1, self.skip_rate)

            tmp_data_idx_1 = [key] * len(valid_frames)
            tmp_data_idx_2 = list(valid_frames)

            self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

            # trainData[(subj, action, subact, 'even')] = action_sequence

          if len(completeData) == 0:
            completeData = copy.deepcopy(action_sequence)
          else:
            completeData = np.append(completeData, action_sequence, axis=0)

          key += 1

    # return trainData, completeData
    # print(len(self.data_idx))
    # print(self.data_idx[100])

  def _init_meta(self, opt):
    self.path_to_dataset = opt[self._name]["data_dir"]
    self.one_hot = opt[self._name]["one_hot"]
    self.sample_rate = opt[self._name]["sample_rate"]
    self.skip_rate = opt[self._name]["skip_rate"]

    self.in_n = opt[self._name]["input_n"]
    self.out_n = opt[self._name]["output_n"]

    if self._is_for == "train":
      self._subjects = self._opt[self._name]["train_ids"]
    elif self._is_for == "val":
      self._subjects = self._opt[self._name]["val_ids"]
    elif self._is_for == "test":
      self._subjects = self._opt[self._name]["test_ids"]

    self._actions = self._opt[self._name]["actions"]

    self.p3d = {}
    self.data_idx = []


  def readCSVasFloat(self, filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34
    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
      line = line.strip().split(',')
      if len(line) > 0:
        returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray

  def change_reference(self, sequence, origin):
    first_pose = np.copy(np.expand_dims(sequence[0], 0))
    first_pose -= origin
    delta_array = np.diff(sequence, axis=0)
    output = np.concatenate((first_pose, delta_array))
    print(output[0])
    print(output[1])
    print(output[2])
    output = np.cumsum(output, axis=0)

    return output

  def __getitem__(self, item):
    key, start_frame = self.data_idx[item]
    fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)

    xyz = self.p3d[key][fs]  # [:, self._dimensions_to_use]

    return {'xyz': xyz,
            'target': xyz[-self.out_n:]}

  def __len__(self):
    return len(self.data_idx)


if __name__ == "__main__":
    import sys
    from src.options.config_parser import ConfigParser
    from src.utils.data_utils import iri_discretize_pose,  iri_undiscretize_pose
    from src.utils.plots import animate_h36m_sequence
    import torch

    # np.set_printoptions(threshold=sys.maxsize)

    opt = ConfigParser().get_config()
    subjects = opt['h36m']['train_ids']
    train_subjects = subjects[0]
    actions = opt['h36m']['actions']
    train_dataset = H36M(opt, is_for="train", subset=[], transform=[])
    sample = train_dataset[5000]
    print(sample['xyz'].shape)
    print(sample['xyz'][0])
    animate_h36m_sequence(sample['xyz'], show=True)