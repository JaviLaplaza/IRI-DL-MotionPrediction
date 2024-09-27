from __future__ import division

import numpy as np
import numpy.random as random
from src.transforms.transforms import TransformBase


class RndSkeletonSequenceFlip(TransformBase):
    """Normalize tensor with given mean std
    """

    def __init__(self, perkey_args, general_args):
        super(RndSkeletonSequenceFlip, self).__init__()
        self._params = self._find_params_for_sample_key(perkey_args, general_args)

    def __call__(self, sample):
        do_flip = random.random() < 0.5
        for sample_key in sample.keys():
            if sample_key in self._params.keys():
                if do_flip:
                    sample[sample_key] = self._flip_skeleton_sequence(sample[sample_key])

                else:
                    pass

        return sample

    def _flip_skeleton_sequence(self, seq):
        assert len(seq.shape) == 2

        L, Jxyz = seq.shape

        seq = np.reshape(seq, (L, int(Jxyz/3), 3))

        seq[:, :, 1:] = -seq[:, :, 1:]

        # assert len(skeleton.shape) == 3

        seq = np.reshape(seq, (L, Jxyz))

        return seq



    def __str__(self):
        return 'RndSkeletonSequenceFlip:' + str(self._params)
