from __future__ import division

import numpy as np
import numpy.random as random
from src.transforms.transforms import TransformBase


class Noise(TransformBase):
    """Normalize tensor with given mean std
    """

    def __init__(self, perkey_args, general_args):
        super(Noise, self).__init__()
        self._params = self._find_params_for_sample_key(perkey_args, general_args)

    def __call__(self, sample):
        for sample_key in sample.keys():
            if sample_key in self._params.keys():
                sample[sample_key] = self._add_noise(sample[sample_key])

            else:
                pass

        return sample

    def _add_noise(self, seq):
        assert len(seq.shape) == 2

        for skeleton in seq:
            for joint in skeleton:
                add_noise = random.random() < 0.8

                if add_noise:
                    noise = random.uniform(-0.03, 0.03)
                    joint += noise

        return seq



    def __str__(self):
        return 'Noise:' + str(self._params)
