from __future__ import division

import numpy as np
import numpy.random as random
from src.transforms.transforms import TransformBase


class Translation(TransformBase):
    """Normalize tensor with given mean std
    """

    def __init__(self, perkey_args, general_args):
        super(Translation, self).__init__()
        self._params = self._find_params_for_sample_key(perkey_args, general_args)

    def __call__(self, sample):
        for sample_key in sample.keys():
            if sample_key in self._params.keys():
                sample[sample_key] = self._add_translation(sample[sample_key])

            else:
                pass

        return sample

    def _add_translation(self, seq):
        assert len(seq.shape) == 2

        add_translation = random.random() < 0.8

        if add_translation:
            translation_x = random.uniform(-2, 2)
            translation_y = 0
            translation_z = random.uniform(-2, 2)
            translation = np.array([translation_x, translation_y, translation_z])

            translation = np.tile(translation, int(seq.shape[-1]/3))
            seq += translation

        return seq



    def __str__(self):
        return 'Translation:' + str(self._params)
