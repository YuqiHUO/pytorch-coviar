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
import csv
import pickle


GOP_SIZE = 12

def cls2int(x):
    return int(x[1:])

def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)

def parse_charades_csv(filename):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']
            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [{'class': x, 'start': float(
                    y), 'end': float(z)} for x, y, z in actions]
            labels[vid] = actions
    # print(labels)
    return labels

def cache(cachefile):
    """ Creates a decorator that caches the result to cachefile """
    def cachedecorator(fn):
        def newf(*args, **kwargs):
            print('cachefile {}'.format(cachefile))
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as f:
                    print("Loading cached result from '%s'" % cachefile)
                    return pickle.load(f)
            res = fn(*args, **kwargs)
            with open(cachefile, 'wb') as f:
                print("Saving result to cache '%s'" % cachefile)
                pickle.dump(res, f)
            return res
        return newf
    return cachedecorator

class CoviarDataSet(data.Dataset):
    def __init__(self, data_root, data_name,
                 video_list,
                 representation,
                 transform,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._data_name = data_name
        self._representation = representation
        self._transform = transform
        self._is_train = is_train
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()


        self.labels = parse_charades_csv(video_list)
        cachename = '{}_{}.pkl'.format(self.__class__.__name__, self._is_train)
        self.data = cache(cachename)(self.prepare)(self.labels)


    def prepare(self, labels):
        FPS= 30
        video_path, gop_index, targets, ids = [], [], [], []

        for i, (vid, label) in enumerate(labels.items()):
            
            vpath = os.path.join(self._data_root, vid + '.mp4')
            num_frames = get_num_frames(vpath)
            num_gop = num_frames // GOP_SIZE
            if self._is_train :
                for x in label:
                    for gop in range(num_gop):
                        if (x['start'] < gop*0.4) and ((gop+1)*0.4 < x['end']):
                            video_path.append(vpath)
                            gop_index.append(gop)
                            targets.append(cls2int(x['class']))
                            ids.append(vid)
            else:
                target = torch.IntTensor(157).zero_()
                for x in label:
                    target[cls2int(x['class'])] = 1
                
                for gop in range(num_gop):
                    video_path.append(vpath)
                    gop_index.append(gop)
                    targets.append(target)
                    ids.append(vid)
                
        print(video_path, gop_index, targets, ids)
        return {'video_path': video_path, 'gop_index': gop_index, 'targets': targets, 'ids': ids}



    def __getitem__(self, index):

        video_path = self.data['video_path'][index]
        gop_index = self.data['gop_index'][index]
        target = self.data['targets'][index]
        # print(video_path, gop_index, target)

        if self._representation == 'iframe':
            frames_i = []
            img_i = load(video_path, gop_index, 0, 0, self._accumulate)
            img_i = color_aug(img_i)
            img_i = img_i[..., ::-1]
            frames_i.append(img_i)
            frames_i = self._transform(frames_i)
            frames_i = np.array(frames_i)
            frames_i = np.transpose(frames_i, (0, 3, 1, 2))
            input_i = torch.from_numpy(frames_i).float() / 255.0
            input_i = (input_i - self._input_mean) / self._input_std
            input = input_i


        if self._representation == 'mv':
            frames_m = []
            img_m = load(video_path, gop_index, 6, 1, self._accumulate)
            img_m = clip_and_scale(img_m, 20)
            img_m += 128
            img_m = (np.minimum(np.maximum(img_m, 0), 255)).astype(np.uint8)
            frames_m.append(img_m)
            frames_m = self._transform(frames_m)
            frames_m = np.array(frames_m)
            frames_m = np.transpose(frames_m, (0, 3, 1, 2))
            input_m = torch.from_numpy(frames_m).float() / 255.0
            input_m = (input_m - 0.5)
            input = input_m


        if self._representation == 'r':
            frames_r = []                        
            img_r = load(video_path, gop_index, 6, 2, self._accumulate)
            img_r += 128
            img_r = (np.minimum(np.maximum(img_r, 0), 255)).astype(np.uint8)           
            frames_r.append(img_r)      
            frames_r = self._transform(frames_r)     
            frames_r = np.array(frames_r)        
            frames_r = np.transpose(frames_r, (0, 3, 1, 2))          
            input_r = torch.from_numpy(frames_r).float() / 255.0            
            input_r = (input_r - 0.5) / self._input_std
            input = input_r

        # print(input.shape)
        # target = target.long()
        # print(target)
        return input, target


    def __len__(self):
        return len(self.data['video_path'])
