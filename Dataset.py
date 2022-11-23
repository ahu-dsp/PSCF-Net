import os
from PIL.Image import Image
from osgeo import gdal
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from opt import opt
import scipy.io as io
from input_prepocessing import input_preparation, resize_images
from sensor import Sensor


class dataset_all(data.Dataset):
    def __init__(self,filepath):
        super(dataset_all, self).__init__()
        self.filepath_pan = filepath+'/PAN'
        self.lis_pan = os.listdir(self.filepath_pan)
        self.lis_pan.sort()
        self.filepath_ms = filepath+'/MS'
        self.lis_ms = os.listdir(self.filepath_ms)
        self.lis_ms.sort()
        self.filepath_target = filepath+'/target'
        self.lis_target = os.listdir(self.filepath_target)
        self.lis_target.sort()


    def __getitem__(self ,idx):


        pan = gdal.Open(self.filepath_pan + '/' + self.lis_pan[idx])
        ms = gdal.Open(self.filepath_ms + '/' + self.lis_ms[idx])
        target = gdal.Open( self.filepath_target + '/' + self.lis_target[idx])
        ms = ms.ReadAsArray()
        pan = pan.ReadAsArray()  # tif to np
        target = target.ReadAsArray()
        target = target.astype('float64')
        pan = np.expand_dims(pan ,axis=0)
        ms = ms / (2 ** 11)
        pan = pan / (2 ** 11)

        I_ms = torch.from_numpy(ms).float()
        I_pan = torch.from_numpy(pan).float()
        I_MS_target = torch.from_numpy(target).float()

        # temp = io.loadmat(self.filepath_mat + '/' + self.lis[idx])
        # s = Sensor(sensor)
        # I_PAN = temp['I_PAN'].astype('float32')
        # I_MS = temp['I_MS_LR'].astype('float32')
        # # I_MS_target = I_MS.astype('uint8')
        # I_MS2 ,I_PAN2 = resize_images(I_MS ,I_PAN ,s.ratio ,s.sensor)
        # I_ms ,I_pan = input_preparation(I_MS2 ,I_PAN2 ,s.ratio ,s.nbits ,s.net_scope)
        # transform = Compose([ToTensor(), Stretch()])
        # I_ms = transform(I_ms)
        # I_pan = transform(I_pan)
        # I_MS_target = transform(I_MS)
        # I_ms = I_ms.type(torch.FloatTensor)
        # I_pan = I_pan.type(torch.FloatTensor)
        # I_MS_target = I_MS_target.type(torch.FloatTensor)
        # print(ms.shape)
        return I_pan,I_ms,I_MS_target

    def __len__(self):
        return len(self.lis_ms)



