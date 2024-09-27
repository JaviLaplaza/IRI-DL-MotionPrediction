from __future__ import print_function
from PIL import Image
import os
import torchvision
import math
import torch
import numpy as np
# import cv2


def tensor2im(img, imtype=np.uint8, unnormalize=True, nrows=None, to_numpy=False):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = torchvision.utils.make_grid(img, nrow=nrows)

    # unnormalize
    img = img.cpu().float()
    if unnormalize:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)
        img *= 255

    # to numpy
    image_numpy = img.numpy()
    if to_numpy:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy.astype(imtype)


def resize_numpy_tensor(tensor, scale, interpolation):
    tensor = tensor.transpose((1, 2, 0))
    # tensor = cv2.resize(tensor, None, fx=scale, fy=scale, interpolation=interpolation)
    tensor = tensor.transpose((2, 0, 1))
    return tensor


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(image_numpy, image_path):
    mkdir(os.path.dirname(image_path))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_str_data(data, path):
    mkdir(os.path.dirname(path))
    np.savetxt(path, data, delimiter=",", fmt="%s")


def append_dictionaries(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = [v]
        else:
            dict1[k].append(v)
    return dict1

def mean_dictionary(dict):
    for k, v in dict.items():
        dict[k] = np.mean(np.array(v))
    return dict

def lr_decay_mine(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def orth_project(cam, pts):
    """

    :param cam: b*[s,tx,ty]
    :param pts: b*k*3
    :return:
    """
    s = cam[:, 0:1].unsqueeze(1).repeat(1, pts.shape[1], 2)
    T = cam[:, 1:].unsqueeze(1).repeat(1, pts.shape[1], 1)

    return torch.mul(s, pts[:, :, :2] + T)


def opt_cam(x, x_target):
    """
    :param x: N K 3 or  N K 2
    :param x_target: N K 3 or  N K 2
    :return:
    """
    if x_target.shape[2] == 2:
        vis = torch.ones_like(x_target[:, :, :1])
    else:
        vis = (x_target[:, :, :1] > 0).float()
    vis[:, :2] = 0
    xxt = x_target[:, :, :2]
    xx = x[:, :, :2]
    x_vis = vis * xx
    xt_vis = vis * xxt
    num_vis = torch.sum(vis, dim=1, keepdim=True)
    mu1 = torch.sum(x_vis, dim=1, keepdim=True) / num_vis
    mu2 = torch.sum(xt_vis, dim=1, keepdim=True) / num_vis
    xmu = vis * (xx - mu1)
    xtmu = vis * (xxt - mu2)

    eps = 1e-6 * torch.eye(2).float().cuda()
    Ainv = torch.inverse(torch.matmul(xmu.transpose(1, 2), xmu) + eps.unsqueeze(0))
    B = torch.matmul(xmu.transpose(1, 2), xtmu)
    tmp_s = torch.matmul(Ainv, B)
    scale = ((tmp_s[:, 0, 0] + tmp_s[:, 1, 1]) / 2.0).unsqueeze(1)

    scale = torch.clamp(scale, 0.7, 10)
    trans = mu2.squeeze(1) / scale - mu1.squeeze(1)
    opt_cam = torch.cat([scale, trans], dim=1)
    return opt_cam


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2. / N)
            if k == 0:
                w = np.sqrt(1. / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1. / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m
