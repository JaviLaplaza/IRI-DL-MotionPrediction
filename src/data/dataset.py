import torch.utils.data as data
import os
import os.path


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, *args, **kwargs):
        print(dataset_name)
        if dataset_name == 'cifar10':
            from src.data.cifar10 import Cifar10Dataset
            dataset = Cifar10Dataset(*args, **kwargs)
        elif dataset_name == 'mediapipe_handover':
            from src.data.mediapipe_handover import MediapipeHandover
            dataset = MediapipeHandover(*args, **kwargs)
        elif dataset_name == 'h36m':
            from src.data.h36m import H36M
            dataset = H36M(*args, **kwargs)
        elif dataset_name == 'canopies':
            from src.data.canopies import CANOPIESDataset
            dataset = CANOPIESDataset(*args, **kwargs)
        elif dataset_name == 'assistive_robotics':
            from src.data.assistive_robotics import AssistiveRoboticsDataset
            dataset = AssistiveRoboticsDataset(*args, **kwargs)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} was created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, opt, is_for, subset, transform, dataset_type):
        if not hasattr(self, '_name'):
            self._name = 'DatasetBase'
        super(DatasetBase, self).__init__()
        self._opt = opt
        self._is_for = is_for
        self._subset = subset
        self._transform = transform
        self._dataset_type = dataset_type

        self._IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def get_transform(self):
        return self._transform

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self._IMG_EXTENSIONS)

    def _is_csv_file(self, filename):
        return filename.endswith('.csv')

    def _get_all_files_in_subfolders(self, dir, is_file):
        files = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname):
                    path = os.path.join(root, fname)
                    path = os.path.relpath(path, dir)
                    files.append(path)

        return files