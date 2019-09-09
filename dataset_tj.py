"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors, 
or residuals) for training or testing.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data

from coviar import get_num_frames
from coviar import load
from transforms import color_aug


GOP_SIZE = 12


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)

class CoviarDataSet(data.Dataset):
    def __init__(self, data_root, data_name,
                 transform_i, 
                 transform_m, 
                 transform_r,
                 video_list,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._data_name = data_name
        self._transform_i = transform_i
        self._transform_m = transform_m
        self._transform_r = transform_r
        self._is_train = is_train
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        with open(video_list, 'r') as f:
            for line in f:
                video, _, label = line.strip().split()
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                self._video_list.append((
                    video_path,
                    int(label),
                    get_num_frames(video_path)))

        print('%d videos loaded.' % len(self._video_list))

    def __getitem__(self, index):

        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        
        num_gop = num_frames // GOP_SIZE

        for gop in range(num_gop):

            frames_i = []
            frames_m = []
            frames_r = []

            img_i = load(video_path, gop, 0, 0, self._accumulate)
            img_i = color_aug(img_i)
            img_i = img_i[..., ::-1]

            img_m = load(video_path, gop, 6, 1, self._accumulate)
            img_m = clip_and_scale(img_m, 20)
            img_m += 128
            img_m = (np.minimum(np.maximum(img_m, 0), 255)).astype(np.uint8)

            img_r = load(video_path, gop, 6, 2, self._accumulate)
            img_r += 128
            img_r = (np.minimum(np.maximum(img_r, 0), 255)).astype(np.uint8)

            frames_i.append(img_i)
            frames_m.append(img_m)
            frames_r.append(img_r)

            frames_i = self._transform_i(frames_i)
            frames_m = self._transform_m(frames_m)
            frames_r = self._transform_r(frames_r)

            frames_i = np.array(frames_i)
            frames_m = np.array(frames_m)
            frames_r = np.array(frames_r)
            frames_i = np.transpose(frames_i, (0, 3, 1, 2))
            frames_m = np.transpose(frames_m, (0, 3, 1, 2))
            frames_r = np.transpose(frames_r, (0, 3, 1, 2))
            input_i = torch.from_numpy(frames_i).float() / 255.0
            input_m = torch.from_numpy(frames_m).float() / 255.0
            input_r = torch.from_numpy(frames_r).float() / 255.0

            input_i = (input_i - self._input_mean) / self._input_std
            input_m = (input_m - 0.5)
            input_r = (input_r - 0.5) / self._input_std

            # print(input_i.shape)
            # a=input_i.view((-1, ) + input_i.size()[-3:])
            # print(a.shape)
            # print(input_m.shape)
            # print(input_r.shape)

            input1 = torch.cat((input_i, input_m, input_r), 1)

            # print(input1.shape)

            if gop == 0:
                input = input1
            else:
                input = torch.cat((input, input1), 0)
        # print(input.shape)
        # a=input.view((-1, ) + input.size()[-3:])
        # print(a.shape)
        # print(input)
        return input, label

    def __len__(self):
        return len(self._video_list)
