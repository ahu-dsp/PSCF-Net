import os
from PIL.Image import Image
from matplotlib import pyplot as plt
from osgeo import gdal
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from opt import opt
from skimage import io
from sensor import Sensor
from input_prepocessing import input_preparation, resize_images
from skimage.transform import resize
from interpolator_tools import interp23tap
from util import show


filepath = 'test/WV'
filepath_pan = filepath + '/新建文件夹'
lis_pan = os.listdir(filepath_pan)
lis_pan.sort()
filepath_ms = filepath + '/target'
lis_ms = os.listdir(filepath_ms)
lis_ms.sort()
sensor = 'WV2'
s = Sensor(sensor)
a=0

for idx in range(len(lis_pan)):#len(lis_pan):
    pan = gdal.Open(filepath_pan + '/' + lis_pan[idx])
    ms = gdal.Open(filepath_ms + '/' + lis_ms[idx])
    ms = ms.ReadAsArray()
    pan = pan.ReadAsArray()  # tif to np

    ms2 = ms

    ms = np.moveaxis(ms ,0 ,2)
    ms = ms.astype('float32')
    pan = pan.astype('float32')
    I_MS ,I_PAN = resize_images(ms ,pan ,s.ratio ,s.sensor)
    I_ms ,I_pan = input_preparation( I_MS ,I_PAN  ,s.ratio ,s.nbits ,s.net_scope)
    I_ms = np.moveaxis(I_ms ,2 ,0)
    # I_pan = np.moveaxis(I_pan ,2 ,0)

    # print(filepath_pan + '/' + lis_pan[idx])
    # print(I_ms.shape)
    # print(I_pan.shape)


    name_pan = 'test/WV/PAN' + '/' + lis_pan[idx]
    name_ms = 'test/WV/MS' + '/' + lis_ms[idx]
    # print(I_ms)
    # print(I_PAN)
    io.imsave(name_ms ,I_ms)
    io.imsave(name_pan ,I_pan)
    # a=io.imread(name_pan)
    print(a+1)
    a+=1


# for idx in range(len(lis_pan)):#len(lis_pan):
#     pan = gdal.Open(filepath_pan + '/' + lis_pan[idx])
#     ms = gdal.Open(filepath_ms + '/' + lis_ms[idx])
#     ms = ms.ReadAsArray()
#     pan = pan.ReadAsArray()  # tif to np
#
#     ms = np.moveaxis(ms ,0 ,2)
#     ms = ms.astype('float32')
#     pan = pan.astype('float32')
#     # I_MS ,I_PAN = resize_images(ms ,pan ,s.ratio ,s.sensor)
#     I_ms ,I_pan = input_preparation(ms ,pan ,s.ratio ,s.nbits ,s.net_scope)
#     I_ms = np.moveaxis(I_ms ,2 ,0)
#     # I_pan = np.moveaxis(I_pan ,2 ,0)
#
#     # print(filepath_pan + '/' + lis_pan[idx])
#     # print(I_ms.shape)
#     # print(I_pan.shape)
#     name_ms = '32/ms4'+ '/' + lis_ms[idx]
#     # name_pan = 'train/4' + '/' + lis_pan[idx]
#     # print(I_ms)
#     # print(I_PAN)
#     io.imsave(name_ms ,I_ms)
#     # io.imsave(name_pan ,I_pan)
#     # a=io.imread(name_pan)
#     print(a+1)
#     a+=1




