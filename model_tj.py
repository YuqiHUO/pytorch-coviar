"""Model definition."""

import torch
from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision
from train_options import parser

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, num_class):
        super(Model, self).__init__()

        print(("""
Initializing model:
    num_class:          {}.
        """.format(num_class)))

        self._prepare_base_model()
        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):

        # feature_dim_i = getattr(self.base_model_i, 'fc').in_features
        # feature_dim_m = getattr(self.base_model_m, 'fc').in_features
        # feature_dim_r = getattr(self.base_model_r, 'fc').in_features
        # setattr(self.base_model_i, 'fc', nn.Linear(feature_dim_i, num_class))
        # setattr(self.base_model_m, 'fc', nn.Linear(feature_dim_m, num_class))
        # setattr(self.base_model_r, 'fc', nn.Linear(feature_dim_r, num_class))
        
        # resnet
        setattr(self.base_model_m, 'conv1',
                    nn.Conv2d(2, 64, 
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False))
        self.data_bn_m = nn.BatchNorm2d(2)
        
        self.data_bn_r = nn.BatchNorm2d(3)

        self.base_model_i = nn.Sequential(*list(self.base_model_i.children())[:-1])
        self.base_model_m = nn.Sequential(*list(self.base_model_m.children())[:-1])
        self.base_model_r = nn.Sequential(*list(self.base_model_r.children())[:-1])

        # lstm
        mysize=3072
        self.linear_1=torch.nn.Linear(mysize,num_class)
        

        self.lstm = nn.LSTM(input_size=mysize,  #输入数据的特征数是4
                hidden_size=mysize, #输出的特征数（hidden_size）是10
                batch_first= True,
                num_layers=1)
        


    def _prepare_base_model(self):

        self.base_model_i = getattr(torchvision.models, 'resnet152')(pretrained=True)
        #self.base_model_i = nn.Sequential(*list(self.base_model_i.feature.children())[:-2])

        self.base_model_m = getattr(torchvision.models, 'resnet18')(pretrained=True)
        self.base_model_r = getattr(torchvision.models, 'resnet18')(pretrained=True)

        
        self._input_size = 224

    def forward(self, input):
        # resnet
        input_i=input[:,:,0:3,:,:]#[64,4,3,224,224]
        input_m=input[:,:,3:5,:,:]#mv
        input_r=input[:,:,5:8,:,:]
        # print(input_i.shape)
        # print(input_m.shape)
        # print(input_r.shape)
        input_i = input_i.view((-1, ) + input_i.size()[-3:])
        input_m = input_m.view((-1, ) + input_m.size()[-3:])
        input_r = input_r.view((-1, ) + input_r.size()[-3:])
        # a=input_i.view((-1, ) + input_i.size()[-4:])
        # print(a.shape)

        input_m = self.data_bn_m(input_m)
        input_r = self.data_bn_r(input_r)

        base_out_i = self.base_model_i(input_i)
        base_out_m = self.base_model_m(input_m)
        base_out_r = self.base_model_r(input_r)
        # print(base_out_i.shape)
        # print(base_out_m.shape)
        # print(base_out_r.shape)

        base_out = torch.cat((base_out_i, base_out_m, base_out_r), 1)#[batchsize*?num_gop, 3072, 1, 1]
        print(base_out.shape)

        # lstm
        args = parser.parse_args()
        input_l = base_out.view((args.batch_size,-1) + base_out.size()[1:])#>[batchsize,num_gop, 3072, 1, 1]
        print(input_l.shape)
        input_l = torch.squeeze(input_l, 3)#[batchsize,num_gop, 3072, 1]
        print(input_l.shape)
        input_l = torch.squeeze(input_l, 3)#[batchsize,num_gop, 3072]
        print(input_l.shape)

        _,(input_l,_)=self.lstm(input_l)
        lstm_out=input_l[-1,:,:]
        print(lstm_out.shape)
        lstm_out=lstm_out.view(-1,lstm_out.size()[-1])
        print(lstm_out.shape)

        lstm_out=self.linear_1(lstm_out)
        print(lstm_out.shape)


        return lstm_out

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        scales_i = [1, .875, .75, .66]
        scales_m = [1, .875, .75]
        scales_r = [1, .875, .75]

        print('Augmentation scales_i:', scales_i)
        print('Augmentation scales_m:', scales_m)
        print('Augmentation scales_r:', scales_r)

        transform_i = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales_i),
             GroupRandomHorizontalFlip(is_mv=False)])
        transform_m = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales_m),
             GroupRandomHorizontalFlip(is_mv=True)])
        transform_r = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales_r),
             GroupRandomHorizontalFlip(is_mv=False)])

        return transform_i, transform_m, transform_r
