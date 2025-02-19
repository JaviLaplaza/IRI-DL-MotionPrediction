import argparse

import torch
from torchvision import transforms

from src.options.config_parser import ConfigParser
from src.data.custom_dataset_data_loader import CustomDatasetDataLoader
from src.models.models import ModelsFactory
from src.utils.util import mkdir, tensor2im
from tqdm import tqdm
import time
import os
import imageio
import glob
import cv2
import numpy as np

import torchvision

class Test:
    def __init__(self, args):
        config_parser = ConfigParser(set_master_gpu=False)
        self._opt = config_parser.get_config()
        self._opt["model"]["is_train"] = False

        # prepare data
        self._prepare_data()

        # check options
        self._check_options()

        # Set master gpu
        self._set_gpus(args.gpu, config_parser)

        # set output dir
        self._set_output()

        # create model
        model_type = self._opt["model"]["type"]
        self._model = ModelsFactory.get_by_name(model_type, self._opt)

        # test
        self._test_dataset()

    def _prepare_data(self):
        data_loader_test = CustomDatasetDataLoader(self._opt, is_for="test")
        self._dataset_test = data_loader_test.load_data()
        self._dataset_test_size = len(data_loader_test)
        self._train_batch_size = data_loader_test.get_batch_size()
        print(f'#test images = {self._dataset_test_size}')

    def _check_options(self):
        assert self._opt["dataset_test"]["batch_size"] == 1
        assert self._opt["dataset_test"]["serial_batches"]

    def _set_gpus(self, gpu, config_parser):
        if gpu != -1:
            self._opt["misc"]["master_gpu"] = args.gpu
            self._opt["misc"]["G_gpus"] = [args.gpu]
        config_parser.set_gpus()

    def _set_output(self):
        self._save_foler = time.strftime("%d_%m_%H_%M_%S")
        mkdir(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], self._save_foler))

    def _test_dataset(self):
        self._model.set_eval()

        total_time = 0
        n_total_time = 0
        for i_test_batch, test_batch in tqdm(enumerate(self._dataset_test), total=len(self._dataset_test)):
            # set inputs
            self._model.set_input(test_batch)

            # get estimate
            start_wait = time.time()
            estimate = self._model.evaluate()
            total_time += time.time() - start_wait
            n_total_time += 1

            # store estimate
            self._save_seq(estimate, i_test_batch, fps=30)

        print(f"mean time per sample: {total_time/n_total_time}")

    def _save_seq(self, seq, id, fps):
        filename = "{0:05d}.png".format(id)
        filepath = os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"], self._save_foler, filename)
        # seq = torch.permute(seq, [0, 2, 3, 1]).to(dtype=torch.uint8)
        im_seq = []
        # videodims = (640, 480)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # video = cv2.VideoWriter(filepath, fourcc, 60, videodims)
        # torchvision.io.write_video(filename=filepath, video_array=seq, fps=fps)
        for frame in seq:
            torchvision.io.write_jpeg(frame.to(dtype=torch.uint8), filename=filepath)
            # im_seq.append(transforms.ToPILImage()(frame).convert("RGB"))
            # video.write(cv2.cvtColor(np.array(transforms.ToPILImage()(frame).convert("RGB")), cv2.COLOR_RGB2BGR))
            # video.write(cv2.cvtColor(np.array(transforms.ToPILImage()(frame)), cv2.COLOR_RGB2BGR))

        # video.release()
        # torchvision.io.write_video(filename=filepath, video_array=im_seq, fps=fps, video_codec='libx264')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to run test')
    args, _ = parser.parse_known_args()
    Test(args)
