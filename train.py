import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from osgeo import gdal
from sensor import Sensor
import numpy as np
from tensorboardX import SummaryWriter
import Dataset
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import Model
from torch import nn,optim
import torch.nn as nn
from tqdm import tqdm
import time
from loss import ssim,ms_ssim,MS_SSIM
from metrics import scc
from metrics import ergas
from metrics import  sam2 as sam
from metrics import QAVE,PSNR,cross_correlation,RMSE,Q2n,ReproQ2n
from util import save_figure, visualize_tensor, avg_metric,show
from opt import opt
from input_prepocessing import input_preparation, resize_images
from skimage import io

def main():

    batch_size = opt.BatchSize
    lr = opt.lr
    filepath = opt.filepath
    Epoch = opt.epoch
    train =opt.train
    pre_train = opt.pre_train

    if pre_train == True:
        pretrain_model = opt.pretrain_model

    if train == True:
        dataset = Dataset.dataset_all(filepath=filepath)
        dataloader = data.DataLoader(dataset=dataset ,batch_size=batch_size ,shuffle=True ,num_workers=0 ,
                                     drop_last=True)

        print("===> Setting Model")
        if pre_train == False:

            Module = Model.PSCF_Net()

            print('just train')

        else:
            Module = torch.load(pretrain_model)
            print('pretrain_model train')
        # print(Module)
        print("===> Setting GPU")

        if torch.cuda.is_available():
            Module = Module.cuda()


        print("===> Setting optimizer")

        optimizer = optim.Adam(Module.parameters() ,lr=lr)

        print("===> Setting loss")
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()
        criterion3 = nn.SmoothL1Loss()
        Module.train()
        Loss_list = []
        bar = tqdm(range(Epoch))

        print("===> Training")
        writer = SummaryWriter("logs")
        loss_step=0
        total_test_step = 0
        save = 0
        for epoch in bar:
            print("---------第{}轮训练开始---------".format(epoch + 1))
            start_time = time.time()
            for pan,MS,traget in dataloader:
                save+=1
                loss_list=[]
                img_pan = pan
                img_MS = MS
                img_target = traget

                if torch.cuda.is_available():
                    img_pan = img_pan.cuda()
                    img_MS = img_MS.cuda()
                    img_target = img_target.cuda()

                img_out = Module(img_pan,img_MS)


                loss = criterion3(img_out,img_target)+250*(1-ssim(img_out,img_target))

                loss_list.append(loss.item())
                writer.add_scalar('loss' ,loss.item() ,loss_step)
                bar.set_description("Epoch: %d    Loss: %.6f" % (epoch+1 ,loss_list[-1]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_step+=1

            if (epoch+1) % 1 == 0:
            #     if (save+1) % 500 == 0:

                model_name = "model/IKONOS_CFB=4_epoch{}--lr={}.pth".format(epoch ,lr)

                torch.save(Module ,model_name)
                print("model save")
                testmodel = torch.load(model_name)
                if torch.cuda.is_available():
                    testmodel = testmodel.cuda()
                print('read model')
                print(model_name)
                testmodel.eval()
                with torch.no_grad():
                    sensor = opt.sensor
                    if sensor == 'IKONOS':
                        Validation_pan = 'Validation/IKONOS/PAN'
                        lis_pan = os.listdir(Validation_pan)
                        lis_pan.sort()
                        Validation_ms = 'Validation/IKONOS/MS'
                        lis_ms = os.listdir(Validation_ms)
                        lis_ms.sort()
                        Validation_target = 'Validation/IKONOS/target'
                        lis_target = os.listdir(Validation_target)
                        lis_target.sort()
                    elif sensor=='WV2':
                        Validation_pan = 'Validation/WV2/PAN'
                        lis_pan = os.listdir(Validation_pan)
                        lis_pan.sort()
                        Validation_ms = 'Validation/WV2/MS'
                        lis_ms = os.listdir(Validation_ms)
                        lis_ms.sort()
                        Validation_target = 'Validation/WV2/target'
                        lis_target = os.listdir(Validation_target)
                        lis_target.sort()
                    SAM = []
                    sCC = []
                    Ergas = []

                    for idx in range(len(lis_ms)):
                        pan = gdal.Open(Validation_pan + '/' + lis_pan[idx])
                        ms = gdal.Open(Validation_ms + '/' + lis_ms[idx])
                        target = gdal.Open(Validation_target + '/' + lis_target[idx])
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

                        I_ms = I_ms.cuda()
                        I_pan = I_pan.cuda()
                        I_MS_target = I_MS_target.cuda()

                        I_ms = torch.unsqueeze(I_ms ,dim=0)
                        I_pan = torch.unsqueeze(I_pan ,dim=0)
                        I_MS_target = torch.unsqueeze(I_MS_target ,dim=0)

                        img_out = testmodel(I_pan ,I_ms)

                        img3 = img_out
                        img = I_MS_target
                        sam_val = avg_metric(img3 ,img ,sam)
                        sCC_val = avg_metric(img3 ,img ,scc)
                        ergas_val = avg_metric(img3 ,img ,ergas)

                        SAM_name = 'SAM' + str(idx)
                        SCC_name = 'SCC' + str(idx)
                        ERGAS_name = 'ERGAS' + str(idx)

                        SAM.append(sam_val)
                        sCC.append(sCC_val)
                        Ergas.append(ergas_val)

                        print('sam' ,'sCC' ,'ERGAS' ,sam_val ,sCC_val ,ergas_val)

                        writer.add_scalar(SAM_name ,sam_val ,total_test_step)
                        writer.add_scalar(SCC_name ,sCC_val ,total_test_step)
                        writer.add_scalar(ERGAS_name ,ergas_val ,total_test_step)

                        total_test_step += 1

                    print('mean:' ,np.mean(SAM) ,np.mean(sCC) ,np.mean(Ergas))

            writer.close()




    else:
        sensor = opt.sensor
        if sensor == 'WV2':
            test_dir = 'FULL RESOLUTION/WV2'
        elif sensor == 'IKONOS':
            test_dir = 'FULL RESOLUTION/IKONOS'
        with torch.no_grad():
            print("===> Testing")
            testtype = opt.testtype
            if testtype == 'NO-REFERENCE METRICS':
                print('===>NO-REFERENCE METRICS')

                # test_dir = '/Share/home/Z21301095/test/test2'

                MS_imgs = os.listdir(test_dir + '/ms/')
                MS_imgs.sort()
                Pan_imgs = os.listdir(test_dir + '/pan/')
                Pan_imgs.sort()

                sensor = opt.sensor
                s = Sensor(sensor)
                model_name = opt.testmodel
                model = torch.load(model_name)

                if torch.cuda.is_available():
                    model = model.cuda()
                model.eval()
                print('evaling....')
                print(len(MS_imgs))
                for idx in range(len(MS_imgs)):  # len(MS_imgs)
                    pan = gdal.Open(test_dir + '/pan/' + Pan_imgs[idx])
                    ms = gdal.Open(test_dir + '/ms/' + MS_imgs[idx])

                    ms = ms.ReadAsArray()
                    pan = pan.ReadAsArray()  # tif to np

                    ms = np.moveaxis(ms ,0 ,2)
                    ms = ms.astype('float32')
                    pan = pan.astype('float32')
                    I_ms ,I_pan = input_preparation(ms ,pan ,s.ratio ,s.nbits ,s.net_scope)
                    ms = np.moveaxis(I_ms ,2 ,0)


                    # name1 = 'FULL RESOLUTION/IKONOS/ms4/' + Pan_imgs[idx]
                    # io.imsave(name1 ,ms)

                    pan = np.expand_dims(pan ,axis=0)

                    ms = ms / (2 ** 11)
                    pan = pan / (2 ** 11)

                    I_ms = torch.from_numpy(ms).float()
                    I_pan = torch.from_numpy(pan).float()


                    I_ms = I_ms.cuda()
                    I_pan = I_pan.cuda()


                    I_ms = torch.unsqueeze(I_ms ,dim=0)
                    I_pan = torch.unsqueeze(I_pan ,dim=0)

                    img_out = model(I_pan ,I_ms)


                    out = img_out.cpu().detach().numpy()
                    out = np.squeeze(out)
                    # out = np.moveaxis(out ,0 ,-1)
                    # out = out.astype = np.clip(out ,0 ,out.max())

                    print(out.shape)
                    # print(out)
                    # name = '/Share/home/Z21301095/test/result/test/' + Pan_imgs[idx]

                    if sensor == 'WV2':
                        name = 'FULL RESOLUTION/WV2/result/' + Pan_imgs[idx]
                    elif sensor == 'IKONOS':
                        name = 'FULL RESOLUTION/IKONOS/result/' + Pan_imgs[idx]


                    io.imsave(name ,out)



            else:
                sensor = opt.sensor
                if sensor=='WV2':
                    test_dir = 'test/WV'
                elif sensor=='IKONOS':
                    test_dir = 'test/IKONOS'

                MS_imgs = os.listdir(test_dir + '/MS/')
                MS_imgs.sort()
                Pan_imgs = os.listdir(test_dir + '/PAN/')
                Pan_imgs.sort()
                target_imgs = os.listdir(test_dir+'/target/')
                target_imgs.sort()
                sensor = opt.sensor
                s = Sensor(sensor)
                model_name = opt.testmodel
                model = torch.load(model_name)
                print(model)
                if torch.cuda.is_available():
                    model = model.cuda()
                model.eval()
                SAM = []
                sCC = []
                Ergas = []
                q2n = []
                psnr=[]
                for idx in range(len(MS_imgs)):  #len(MS_imgs)
                    pan = gdal.Open(test_dir + '/PAN/' + Pan_imgs[idx])
                    ms = gdal.Open(test_dir + '/MS/' + MS_imgs[idx])
                    target = gdal.Open(test_dir + '/target/' + MS_imgs[idx])

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

                    I_ms = I_ms.cuda()
                    I_pan = I_pan.cuda()
                    I_MS_target = I_MS_target.cuda()

                    I_ms = torch.unsqueeze(I_ms ,dim=0)
                    I_pan = torch.unsqueeze(I_pan ,dim=0)
                    I_MS_target = torch.unsqueeze(I_MS_target ,dim=0)

                    img_out = model(I_pan,I_ms)


                    psnr_val = PSNR( I_MS_target,img_out)
                    psnr_val = psnr_val.cpu().numpy()


                    sam_val = avg_metric(img_out ,I_MS_target ,sam)
                    sCC_val = avg_metric(img_out ,I_MS_target ,scc)
                    ergas_val = avg_metric(img_out ,I_MS_target ,ergas)
                    qave_val = avg_metric(img_out,I_MS_target,QAVE)
                    rmse_val = avg_metric(img_out,I_MS_target,RMSE)



                    img_out1 = torch.squeeze(img_out)
                    img_out1=img_out1.cpu().detach().numpy()
                    I_MS_target = torch.squeeze(I_MS_target)
                    I_MS_target=I_MS_target.cpu().detach().numpy()

                    Q2n_val = Q2n(img_out1 ,I_MS_target)
                    print('sam' ,'sCC' ,'ERGAS','Qave' ,sam_val ,sCC_val ,ergas_val,qave_val)
                    print('Q2n','psnr','rmse',Q2n_val,psnr_val,rmse_val)

                    SAM.append(sam_val)
                    sCC.append(sCC_val)
                    Ergas.append(ergas_val)
                    q2n.append(Q2n_val)
                    psnr.append(psnr_val)


                    out = img_out.cpu().detach().numpy()
                    out = np.squeeze(out)
                    # out = np.moveaxis(out ,0 ,-1)
                    # out = out.astype = np.clip(out ,0 ,out.max())

                    # print(out.shape)
                    # print(out)
                    name = 'result/test/'+MS_imgs[idx]
                    print(MS_imgs[idx])
                    io.imsave(name,out)

                print('mean:' ,'SAM',np.mean(SAM) ,'sCC',np.mean(sCC) ,'Ergas',np.mean(Ergas),'Q2n',np.mean(q2n),'psnr',np.mean(psnr))






def avg_metric(target, prediction, metric):
    sum = 0
    batch_size = len(target)
    for i in range(batch_size):
        sum += metric(np.transpose(target.data.cpu().numpy()
                                   [i], (1, 2, 0)), np.transpose(prediction.data.cpu().numpy()[i], (1, 2, 0)))
    return sum/batch_size



if __name__ == "__main__":
    main()

