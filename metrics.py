
import cv2
from scipy import ndimage
from scipy.ndimage import sobel

import scipy.ndimage as ft
from skimage.transform.integral import integral_image as integral
from math import ceil, floor, log2

import numpy as np
from skimage import transform
import math
import torch
from torch import nn

import spectral_tools as ut



def sam(img1, img2):
    """SAM for 3D image, shape (H, W, C); uint or float[0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    inner_product = (img1_ * img2_).sum(axis=2)
    img1_spectral_norm = np.sqrt((img1_**2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_**2).sum(axis=2))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
    return np.mean(np.arccos(cos_theta))

def sam2(ms,ps,degs = True):
    result = np.double(ps)
    target = np.double(ms)
    if result.shape != target.shape:
        raise ValueError('Result and target arrays must have the same shape!')

    bands = target.shape[2]
    rnorm = np.sqrt((result ** 2).sum(axis=2))
    tnorm = np.sqrt((target ** 2).sum(axis=2))
    dotprod = (result * target).sum(axis=2)
    cosines = (dotprod / (rnorm * tnorm))
    sam2d = np.arccos(cosines)
    sam2d[np.invert(np.isfinite(sam2d))] = 0.  # arccos(1.) -> NaN
    if degs:
        sam2d = np.rad2deg(sam2d)
    return sam2d[np.isfinite(sam2d)].mean()

def ergas(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real**2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')

# def sam2(ms,ps,degs = True):
#     result = np.double(ps)
#     target = np.double(ms)
#     if result.shape != target.shape:
#         raise ValueError('Result and target arrays must have the same shape!')
#
#     bands = target.shape[2]
#     rnorm = np.sqrt((result ** 2).sum(axis=2))
#     tnorm = np.sqrt((target ** 2).sum(axis=2))
#     dotprod = (result * target).sum(axis=2)
#     cosines = (dotprod / (rnorm * tnorm))
#     sam2d = np.arccos(cosines)
#     sam2d[np.invert(np.isfinite(sam2d))] = 0.  # arccos(1.) -> NaN
#     if degs:
#         sam2d = np.rad2deg(sam2d)
#     return sam2d[np.isfinite(sam2d)].mean()


def RMSE(result, target):
    result = np.double(result)
    target = np.double(target)
    if result.shape != target.shape:
        raise ValueError('result and target arrays must have the same shape!')
    return ((result - target) ** 2).mean() ** 0.5


def ERGAS(result, target, pixratio=0.5):
    result = np.double(result)
    target = np.double(target)
    if result.shape != target.shape:
        raise ValueError('result and target arrays must have the same shape!')

    bands = target.shape[2]
    addends = np.zeros(bands)
    for band in range(bands):
        addends[band] = ((RMSE(result[:, :, band], target[:, :, band])) / (target[:, :, band].mean())) ** 2
    ergas = 100 * pixratio * ((1.0 / bands) * addends.sum()) ** 0.5

    return ergas


def QAVE(result, target):
    result = np.double(result)
    target = np.double(target)
    if result.shape != target.shape:
        raise ValueError('result and target arrays must have the same shape!')

    rmean = result.mean(axis=2)
    tmean = target.mean(axis=2)

    rmean1 = result[:, :, 0] - rmean
    rmean2 = result[:, :, 1] - rmean
    rmean3 = result[:, :, 2] - rmean
    rmean4 = result[:, :, 3] - rmean

    tmean1 = target[:, :, 0] - tmean
    tmean2 = target[:, :, 1] - tmean
    tmean3 = target[:, :, 2] - tmean
    tmean4 = target[:, :, 3] - tmean

    QR = (1 / result.shape[2] - 1) * (rmean1 ** 2 + rmean2 ** 2 + rmean3 ** 2 + rmean4 ** 2)
    QT = (1 / result.shape[2] - 1) * (tmean1 ** 2 + tmean2 ** 2 + tmean3 ** 2 + tmean4 ** 2)
    QRT = (1 / result.shape[2] - 1) * (rmean1 * tmean1 + rmean2 * tmean2 + rmean3 * tmean3 + rmean4 * tmean4)

    QAVE = result.shape[2] * ((QRT * rmean) * tmean) / ((QR + QT) * ((rmean ** 2) + (tmean ** 2)))
    m, n = QAVE.shape
    Q = (1 / (m * n)) * np.sum(np.sum(QAVE))

    return Q

def sCC(ms, ps):
    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')


    return  (np.sum(ps_sobel*ms_sobel)/np.sqrt(np.sum(ps_sobel*ps_sobel))/np.sqrt(np.sum(ms_sobel*ms_sobel)))

def scc(img1, img2):
    """SCC for 2D (H, W)or 3D (H, W, C) image; uint or float[0, 1]"""
    if not  img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    if img1_.ndim == 2:
        return np.corrcoef(img1_.reshape(1, -1), img2_.rehshape(1, -1))[0, 1]
    elif img1_.ndim == 3:
        #print(img1_[..., i].reshape[1, -1].shape)
        #test = np.corrcoef(img1_[..., i].reshape[1, -1], img2_[..., i].rehshape(1, -1))
        #print(type(test))
        ccs = [np.corrcoef(img1_[..., i].reshape(1, -1), img2_[..., i].reshape(1, -1))[0, 1]
               for i in range(img1_.shape[2])]
        return np.mean(ccs)
    else:
        raise ValueError('Wrong input image dimensions.')



def PSNR(H_fuse, H_ref):
    #Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.sum((H_ref_reshaped-H_fuse_reshaped)**2, dim=1)/H_fuse_reshaped.shape[1])

    # Calculating max of H_ref for each band
    max_H_ref, _ = torch.max(H_ref_reshaped, dim=1)

    # Calculating PSNR
    PSNR = torch.nansum(10*torch.log10(torch.div(max_H_ref, rmse)**2))/N_spectral

    return PSNR


def cross_correlation(H_fuse, H_ref):
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating mean value
    mean_fuse = torch.mean(H_fuse_reshaped, 1).unsqueeze(1)
    mean_ref = torch.mean(H_ref_reshaped, 1).unsqueeze(1)

    CC = torch.sum((H_fuse_reshaped- mean_fuse)*(H_ref_reshaped-mean_ref), 1)/torch.sqrt(torch.sum((H_fuse_reshaped- mean_fuse)**2, 1)*torch.sum((H_ref_reshaped-mean_ref)**2, 1))

    CC = torch.mean(CC)
    return CC




def local_cross_correlation(img_1, img_2, half_width):
    """
        Cross-Correlation Field computation.

        Parameters
        ----------
        img_1 : Numpy Array
            First image on which calculate the cross-correlation. Dimensions: H, W
        img_2 : Numpy Array
            Second image on which calculate the cross-correlation. Dimensions: H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation


        Return
        ------
        L : Numpy array
            The cross-correlation map between img_1 and img_2

    """

    w = int(half_width)
    ep = 1e-20

    if (len(img_1.shape)) != 3:
        img_1 = np.expand_dims(img_1, axis=-1)
    if (len(img_2.shape)) != 3:
        img_2 = np.expand_dims(img_2, axis=-1)

    img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
    img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

    img_1_cum = np.zeros(img_1.shape)
    img_2_cum = np.zeros(img_2.shape)
    for i in range(img_1.shape[-1]):
        img_1_cum[:, :, i] = integral(img_1[:, :, i]).astype(np.float64)
    for i in range(img_2.shape[-1]):
        img_2_cum[:, :, i] = integral(img_2[:, :, i]).astype(np.float64)

    img_1_mu = (img_1_cum[2 * w:, 2 * w:, :] - img_1_cum[:-2 * w, 2 * w:, :] - img_1_cum[2 * w:, :-2 * w, :]
                + img_1_cum[:-2 * w, :-2 * w, :]) / (4 * w ** 2)
    img_2_mu = (img_2_cum[2 * w:, 2 * w:, :] - img_2_cum[:-2 * w, 2 * w:, :] - img_2_cum[2 * w:, :-2 * w, :]
                + img_2_cum[:-2 * w, :-2 * w, :]) / (4 * w ** 2)

    img_1 = img_1[w:-w, w:-w, :] - img_1_mu
    img_2 = img_2[w:-w, w:-w, :] - img_2_mu

    img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
    img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

    i2 = img_1 ** 2
    j2 = img_2 ** 2
    ij = img_1 * img_2

    i2_cum = np.zeros(i2.shape)
    j2_cum = np.zeros(j2.shape)
    ij_cum = np.zeros(ij.shape)

    for i in range(i2_cum.shape[-1]):
        i2_cum[:, :, i] = integral(i2[:, :, i]).astype(np.float64)
    for i in range(j2_cum.shape[-1]):
        j2_cum[:, :, i] = integral(j2[:, :, i]).astype(np.float64)
    for i in range(ij_cum.shape[-1]):
        ij_cum[:, :, i] = integral(ij[:, :, i]).astype(np.float64)

    sig2_ij_tot = (ij_cum[2 * w:, 2 * w:, :] - ij_cum[:-2 * w, 2 * w:, :] - ij_cum[2 * w:, :-2 * w, :]
                   + ij_cum[:-2 * w, :-2 * w, :])
    sig2_ii_tot = (i2_cum[2 * w:, 2 * w:, :] - i2_cum[:-2 * w, 2 * w:, :] - i2_cum[2 * w:, :-2 * w, :]
                   + i2_cum[:-2 * w, :-2 * w, :])
    sig2_jj_tot = (j2_cum[2 * w:, 2 * w:, :] - j2_cum[:-2 * w, 2 * w:, :] - j2_cum[2 * w:, :-2 * w, :]
                   + j2_cum[:-2 * w, :-2 * w, :])

    sig2_ij_tot = np.clip(sig2_ij_tot, ep, sig2_ij_tot.max())
    sig2_ii_tot = np.clip(sig2_ii_tot, ep, sig2_ii_tot.max())
    sig2_jj_tot = np.clip(sig2_jj_tot, ep, sig2_jj_tot.max())

    xcorr = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return xcorr


def normalize_block(im):
    """
        Auxiliary Function for Q2n computation.

        Parameters
        ----------
        im : Numpy Array
            Image on which calculate the statistics. Dimensions: H, W

        Return
        ------
        y : Numpy array
            The normalized version of im
        m : float
            The mean of im
        s : float
            The standard deviation of im

    """

    m = np.mean(im)
    s = np.std(im, ddof=1)

    if s == 0:
        s = 1e-10

    y = ((im - m) / s) + 1

    return y, m, s


def cayley_dickson_property_1d(onion1, onion2):
    """
        Cayley-Dickson construction for 1-D arrays.
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        onion1 : Numpy Array
            First 1-D array
        onion2 : Numpy Array
            Second 1-D array

        Return
        ------
        ris : Numpy array
            The result of Cayley-Dickson construction on the two arrays.
    """

    n = onion1.__len__()

    if n > 1:
        half_pos = int(n / 2)
        a = onion1[:half_pos]
        b = onion1[half_pos:]

        neg = np.ones(b.shape)
        neg[1:] = -1

        b = b * neg
        c = onion2[:half_pos]
        d = onion2[half_pos:]
        d = d * neg

        if n == 2:
            ris = np.concatenate([(a * c) - (d * b), (a * d) + (c * b)])
        else:
            ris1 = cayley_dickson_property_1d(a, c)

            ris2 = cayley_dickson_property_1d(d, b * neg)
            ris3 = cayley_dickson_property_1d(a * neg, d)
            ris4 = cayley_dickson_property_1d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.concatenate([aux1, aux2])
    else:
        ris = onion1 * onion2

    return ris


def cayley_dickson_property_2d(onion1, onion2):
    """
        Cayley-Dickson construction for 2-D arrays.
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        onion1 : Numpy Array
            First MultiSpectral img. Dimensions: H, W, Bands
        onion2 : Numpy Array
            Second MultiSpectral img. Dimensions: H, W, Bands

        Return
        ------
        ris : Numpy array
            The result of Cayley-Dickson construction on the two arrays.
    """

    dim3 = onion1.shape[-1]
    if dim3 > 1:
        half_pos = int(dim3 / 2)

        a = onion1[:, :, :half_pos]
        b = onion1[:, :, half_pos:]
        b = np.concatenate([np.expand_dims(b[:, :, 0], -1), -b[:, :, 1:]], axis=-1)

        c = onion2[:, :, :half_pos]
        d = onion2[:, :, half_pos:]
        d = np.concatenate([np.expand_dims(d[:, :, 0], -1), -d[:, :, 1:]], axis=-1)

        if dim3 == 2:
            ris = np.concatenate([(a * c) - (d * b), (a * d) + (c * b)], axis=-1)
        else:
            ris1 = cayley_dickson_property_2d(a, c)
            ris2 = cayley_dickson_property_2d(d,
                                              np.concatenate([np.expand_dims(b[:, :, 0], -1), -b[:, :, 1:]], axis=-1))
            ris3 = cayley_dickson_property_2d(np.concatenate([np.expand_dims(a[:, :, 0], -1), -a[:, :, 1:]], axis=-1),
                                              d)
            ris4 = cayley_dickson_property_2d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.concatenate([aux1, aux2], axis=-1)
    else:
        ris = onion1 * onion2

    return ris


def q_index_metric(im1, im2, size):
    """
        Q2n calculation on a window of dimension (size, size).
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        im1 : Numpy Array
            First MultiSpectral img. Dimensions: H, W, Bands
        im2 : Numpy Array
            Second MultiSpectral img. Dimensions: H, W, Bands
        size : int
            The size of the squared windows on which calculate the UQI index


        Return
        ------
        q : Numpy array
            The Q2n calculated on a window of dimension (size,size).
    """

    im1 = im1.astype(np.double)
    im2 = im2.astype(np.double)
    im2 = np.concatenate([np.expand_dims(im2[:, :, 0], -1), -im2[:, :, 1:]], axis=-1)

    depth = im1.shape[-1]
    for i in range(depth):
        im1[:, :, i], m, s = normalize_block(im1[:, :, i])
        if m == 0:
            if i == 0:
                im2[:, :, i] = im2[:, :, i] - m + 1
            else:
                im2[:, :, i] = -(-im2[:, :, i] - m + 1)
        else:
            if i == 0:
                im2[:, :, i] = ((im2[:, :, i] - m) / s) + 1
            else:
                im2[:, :, i] = -(((-im2[:, :, i] - m) / s) + 1)

    m1 = np.mean(im1, axis=(0, 1))
    m2 = np.mean(im2, axis=(0, 1))

    mod_q1m = np.sqrt(np.sum(m1 ** 2))
    mod_q2m = np.sqrt(np.sum(m2 ** 2))

    mod_q1 = np.sqrt(np.sum(im1 ** 2, axis=-1))
    mod_q2 = np.sqrt(np.sum(im2 ** 2, axis=-1))

    term2 = mod_q1m * mod_q2m
    term4 = mod_q1m ** 2 + mod_q2m ** 2
    temp = (size ** 2) / (size ** 2 - 1)
    int1 = temp * np.mean(mod_q1 ** 2)
    int2 = temp * np.mean(mod_q2 ** 2)
    int3 = temp * (mod_q1m ** 2 + mod_q2m ** 2)
    term3 = int1 + int2 - int3

    mean_bias = 2 * term2 / term4

    if term3 == 0:
        q = np.zeros((1, 1, depth), dtype='float64')
        q[:, :, -1] = mean_bias
    else:
        cbm = 2 / term3
        qu = cayley_dickson_property_2d(im1, im2)
        qm = cayley_dickson_property_1d(m1, m2)

        qv = temp * np.mean(qu, axis=(0, 1))
        q = qv - temp * qm
        q = q * mean_bias * cbm

    return q


def Q2n(outputs, labels, q_block_size=32, q_shift=32):
    """
        Q2n calculation on a window of dimension (size, size).
        Auxiliary function for Q2n calculation.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144
        [Garzelli09]        A. Garzelli and F. Nencini, "Hypercomplex quality assessment of multi/hyper-spectral images,"
                            IEEE Geoscience and Remote Sensing Letters, vol. 6, no. 4, pp. 662-665, October 2009.
        [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods",
                            IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.

        Parameters
        ----------
        outputs : Numpy Array
            The Fused image. Dimensions: H, W, Bands
        labels : Numpy Array
            The reference image. Dimensions: H, W, Bands
        q_block_size : int
            The windows size on which calculate the Q2n index
        q_shift : int
            The stride for Q2n index calculation

        Return
        ------
        q2n_index : float
            The Q2n index.
        q2n_index_map : Numpy Array
            The Q2n map, on a support of (q_block_size, q_block_size)
    """

    height, width, depth = labels.shape
    stepx = ceil(height / q_shift)
    stepy = ceil(width / q_shift)

    if stepy <= 0:
        stepx = 1
        stepy = 1

    est1 = (stepx - 1) * q_shift + q_block_size - height
    est2 = (stepy - 1) * q_shift + q_block_size - width

    if (est1 != 0) and (est2 != 0):
        labels = np.pad(labels, ((0, est1), (0, est2), (0, 0)), mode='reflect')
        outputs = np.pad(outputs, ((0, est1), (0, est2), (0, 0)), mode='reflect')

        outputs = outputs.astype(np.int16)
        labels = labels.astype(np.int16)

    height, width, depth = labels.shape

    if ceil(log2(depth)) - log2(depth) != 0:
        exp_difference = 2 ** (ceil(log2(depth))) - depth
        diff_zeros = np.zeros((height, width, exp_difference), dtype="float64")
        labels = np.concatenate([labels, diff_zeros], axis=-1)
        outputs = np.concatenate([outputs, diff_zeros], axis=-1)

    height, width, depth = labels.shape

    values = np.zeros((stepx, stepy, depth))
    for j in range(stepx):
        for i in range(stepy):
            values[j, i, :] = q_index_metric(
                labels[j * q_shift:j * q_shift + q_block_size, i * q_shift: i * q_shift + q_block_size, :],
                outputs[j * q_shift:j * q_shift + q_block_size, i * q_shift: i * q_shift + q_block_size, :],
                q_block_size
            )

    q2n_index_map = np.sqrt(np.sum(values ** 2, axis=-1))
    q2n_index = np.mean(q2n_index_map)

    return q2n_index.item()


# def ERGAS(outputs, labels, ratio):
#     """
#         Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).
#
#
#         [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
#                             arXiv preprint arXiv:2108.06144
#         [Ranchin00]         T. Ranchin and L. Wald, "Fusion of high spatial and spectral resolution images: the ARSIS concept and its implementation,"
#                             Photogrammetric Engineering and Remote Sensing, vol. 66, no. 1, pp. 4961, January 2000.
#         [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods",
#                             IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
#
#         Parameters
#         ----------
#         outputs : Numpy Array
#             The Fused image. Dimensions: H, W, Bands
#         labels : Numpy Array
#             The reference image. Dimensions: H, W, Bands
#         ratio : int
#             PAN-MS resolution ratio
#
#         Return
#         ------
#         ergas_index : float
#             The ERGAS index.
#
#     """
#
#     mu = np.mean(labels, axis=(0, 1)) ** 2
#     nbands = labels.shape[-1]
#     error = np.mean((outputs - labels) ** 2, axis=(0, 1))
#     ergas_index = 100 / ratio * np.sqrt(np.sum(error / mu) / nbands)
#
#     return np.mean(ergas_index).item()


def SAM(outputs, labels):
    """
        Spectral Angle Mapper (SAM).


        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144
        [Yuhas92]           R. H. Yuhas, A. F. H. Goetz, and J. W. Boardman, "Discrimination among semi-arid landscape endmembers using the Spectral Angle Mapper (SAM) algorithm,"
                            in Proceeding Summaries 3rd Annual JPL Airborne Geoscience Workshop, 1992, pp. 147-149.
        [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods",
                            IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.

        Parameters
        ----------
        outputs : Numpy Array
            The Fused image. Dimensions: H, W, Bands
        labels : Numpy Array
            The reference image. Dimensions: H, W, Bands

        Return
        ------
        angle : float
            The SAM index in degree.
    """

    norm_outputs = np.sum(outputs ** 2, axis=-1)
    norm_labels = np.sum(labels ** 2, axis=-1)
    scalar_product = np.sum(outputs * labels, axis=-1)
    norm_product = np.sqrt(norm_outputs * norm_labels)
    scalar_product[norm_product == 0] = np.nan
    norm_product[norm_product == 0] = np.nan
    scalar_product = scalar_product.flatten()
    norm_product = norm_product.flatten()
    angle = np.sum(np.arccos(np.clip(scalar_product / norm_product, a_min=-1, a_max=1)), axis=-1) / norm_product.shape[0]
    angle = angle * 180 / np.pi

    return angle


def Q(outputs, labels, block_size=32):
    """
        Universal Quality Index (UQI).


        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144
        [Wang02]            Z. Wang and A. C. Bovik, "A universal image quality index,"
                            IEEE Signal Processing Letters, vol. 9, no. 3, pp. 81-84, March 2002.
        [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods",
                            IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.

        Parameters
        ----------
        outputs : Numpy Array
            The Fused image. Dimensions: H, W, Bands
        labels : Numpy Array
            The reference image. Dimensions: H, W, Bands
        block_size : int
            The windows size on which calculate the Q2n index

        Return
        ------
        quality : float
            The UQI index.
    """

    N = block_size ** 2
    nbands = labels.shape[-1]
    kernel = np.ones((block_size, block_size))
    pad_size = floor((kernel.shape[0] - 1) / 2)
    outputs_sq = outputs ** 2
    labels_sq = labels ** 2
    outputs_labels = outputs * labels

    quality = np.zeros(nbands)
    for i in range(nbands):
        outputs_sum = ft.convolve(outputs[:, :, i], kernel)
        labels_sum = ft.convolve(labels[:, :, i], kernel)

        outputs_sq_sum = ft.convolve(outputs_sq[:, :, i], kernel)
        labels_sq_sum = ft.convolve(labels_sq[:, :, i], kernel)
        outputs_labels_sum = ft.convolve(outputs_labels[:, :, i], kernel)
        outputs_sum = outputs_sum[pad_size:-pad_size, pad_size:-pad_size]
        labels_sum = labels_sum[pad_size:-pad_size, pad_size:-pad_size]

        outputs_sq_sum = outputs_sq_sum[pad_size:-pad_size, pad_size:-pad_size]
        labels_sq_sum = labels_sq_sum[pad_size:-pad_size, pad_size:-pad_size]
        outputs_labels_sum = outputs_labels_sum[pad_size:-pad_size, pad_size:-pad_size]

        outputs_labels_sum_mul = outputs_sum * labels_sum
        outputs_labels_sum_mul_sq = outputs_sum ** 2 + labels_sum ** 2

        numerator = 4 * (N * outputs_labels_sum - outputs_labels_sum_mul) * outputs_labels_sum_mul
        denominator_temp = N * (outputs_sq_sum + labels_sq_sum) - outputs_labels_sum_mul_sq
        denominator = denominator_temp * outputs_labels_sum_mul_sq

        index = (denominator_temp == 0) & (outputs_labels_sum_mul_sq != 0)

        quality_map = np.ones(denominator.shape)
        quality_map[index] = 2 * outputs_labels_sum_mul[index] / outputs_labels_sum_mul_sq[index]
        index = denominator != 0
        quality_map[index] = numerator[index] / denominator[index]
        quality[i] = np.mean(quality_map)

    return np.mean(quality).item()


def coregistration(ms, pan, kernel, ratio=4, search_win=4):
    """
        Coregitration function for MS-PAN pair.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        ms : Numpy Array
            The Multi-Spectral image. Dimensions: H, W, Bands
        pan : Numpy Array
            The PAN image. Dimensions: H, W
        kernel : Numpy Array
            The filter array.
        ratio : int
            PAN-MS resolution ratio
        search_win : int
            The windows in which search the optimal value for the coregistration step

        Return
        ------
        r : Numpy Array
            The optimal raw values.
        c : Numpy Array
            The optimal column values.
    """

    nbands = ms.shape[-1]
    p = ft.convolve(pan, kernel, mode='nearest')
    rho = np.zeros((search_win, search_win, nbands))
    r = np.zeros(nbands)
    c = np.copy(r)

    for i in range(search_win):
        for j in range(search_win):
            rho[i, j, :] = np.mean(
                local_cross_correlation(ms, np.expand_dims(p[i::ratio, j::ratio], -1), floor(ratio / 2)), axis=(0, 1))

    max_value = np.amax(rho, axis=(0, 1))

    for b in range(nbands):
        x = rho[:, :, b]
        max_value = x.max()
        pos = np.where(x == max_value)
        if len(pos[0]) != 1:
            pos = (pos[0][0], pos[1][0])
        pos = tuple(map(int, pos))
        r[b] = pos[0]
        c[b] = pos[1]
        r = np.squeeze(r).astype(np.uint8)
        c = np.squeeze(c).astype(np.uint8)

    return r, c


def resize_with_mtf(outputs, ms, pan, sensor, ratio=4, dim_cut=21):
    """
        Resize of Fused Image to MS scale, in according to the coregistration with the PAN.
        If dim_cut is different by zero a cut is made on both outputs and ms, to discard possibly values affected by paddings.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        x : NumPy array
            Fused MultiSpectral image, coregistered with the PAN, low-pass filtered and decimated. If dim_cut is different
            by zero it is also cut
        ms : NumPy array
            MultiSpectral img. If dim_cut is different by zero it is cut.
    """

    from spectral_tools import gen_mtf

    kernel = gen_mtf(ratio, sensor)
    kernel = kernel.astype(np.float32)
    nbands = kernel.shape[-1]
    pad_size = floor((kernel.shape[0] - 1) / 2)

    r, c = coregistration(ms, pan, kernel[:, :, 0], ratio)

    kernel = np.moveaxis(kernel, -1, 0)
    kernel = np.expand_dims(kernel, axis=1)

    kernel = torch.from_numpy(kernel).type(torch.float32)

    depthconv = nn.Conv2d(in_channels=nbands,
                          out_channels=nbands,
                          groups=nbands,
                          kernel_size=kernel.shape,
                          bias=False)
    depthconv.weight.data = kernel
    depthconv.weight.requires_grad = False
    pad = nn.ReplicationPad2d(pad_size)

    x = np.zeros(ms.shape, dtype=np.float32)

    outputs = np.expand_dims(np.moveaxis(outputs, -1, 0), 0)
    outputs = torch.from_numpy(outputs)

    outputs = pad(outputs)
    outputs = depthconv(outputs)

    outputs = outputs.detach().cpu().numpy()
    outputs = np.moveaxis(np.squeeze(outputs, 0), 0, -1)

    for b in range(nbands):
        x[:, :, b] = outputs[r[b]::ratio, c[b]::ratio, b]

    if dim_cut != 0:
        x = x[dim_cut:-dim_cut, dim_cut:-dim_cut, :]
        ms = ms[dim_cut:-dim_cut, dim_cut:-dim_cut, :]

    return x, ms


def ReproERGAS(outputs, ms, pan, sensor, ratio=4, dim_cut=0):
    """
        Reprojected Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        R-ERGAS : float
            The R-ERGAS index

    """

    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    return ERGAS(outputs, ms, ratio)


def ReproSAM(outputs, ms, pan, sensor, ratio=4, dim_cut=0):
    """
        Reprojected Spectral Angle Mapper (SAM).

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        R-SAM : float
            The R-SAM index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    return SAM(outputs, ms)


def ReproQ2n(outputs, ms, pan, sensor, ratio=4, q_block_size=32, q_shift=32, dim_cut=0):
    """
        Reprojected Q2n.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        q_block_size : int
            The windows size on which calculate the Q2n index
        q_shift : int
            The stride for Q2n index calculation
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        r_q2n : float
            The R-Q2n index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    r_q2n, _ = Q2n(outputs, ms, q_block_size, q_shift)
    return r_q2n


def ReproQ(outputs, ms, pan, sensor, ratio=4, q_block_size=32, dim_cut=0):
    """
        Reprojected Q.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        q_block_size : int
            The windows size on which calculate the Q index
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        r_q : float
            The R-Q index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    r_q = Q(outputs, ms, q_block_size)
    return r_q


def ReproMetrics(outputs, ms, pan, sensor, ratio=4, q_block_size=32, q_shift=32, dim_cut=0):
    """
        Computation of all reprojected metrics.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        q_block_size : int
            The windows size on which calculate the Q2n and Q index
        q_shift : int
            The stride for Q2n index calculation
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        r_q2n : float
            The R-Q2n index
        r_q : float
            The R-Q index
        R-SAM : float
            The R-SAM index
        R-ERGAS : float
            The R-ERGAS index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    q2n, _ = Q2n(outputs, ms, q_block_size, q_shift)
    q = Q(outputs, ms, q_block_size)
    sam = SAM(outputs, ms)
    ergas = ERGAS(outputs, ms, ratio)
    return q2n, q, sam, ergas


def DRho(outputs, pan, sigma=4):
    """
        Spatial Quality Index based on local cross-correlation.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sigma : int
            The windows size on which calculate the Drho index; Accordingly with the paper it should be the
            resolution scale which elapses between MS and PAN.

        Return
        ------
        d_rho : float
            The d_rho index

    """
    half_width = ceil(sigma / 2)
    rho = np.clip(local_cross_correlation(outputs, pan, half_width), a_min=-1.0, a_max=1.0)
    d_rho = 1.0 - rho
    return np.mean(d_rho).item()










def gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std)**2) * np.exp(-0.5 * (t2 / std)**2)
    return w

def kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w

def fir_filter_wind(Hd, w):
    """
    compute fir (finite impulse response) filter with window method
    Hd: desired freqeuncy response (2D)
    w: window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)
    return h
def GNyq2win(GNyq, scale=4, N=41):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    scale: spatial size of PAN / spatial size of MS
    """
    #fir filter with window method
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)
    return np.real(h)




def mtf_resize(img, satellite, scale=4):
    # satellite GNyq
    scale = int(scale)
    if satellite == 'WV2':
        GNyq = 0.35 * np.ones((1, 3))
        GNyq = np.append(GNyq, 0.27)
        GNyqPan = np.asarray([0.16])
    elif satellite == 'IKONOS':
        GNyq = [0.26, 0.28, 0.29, 0.28]  # Band Order: B,G,R,NIR
        GNyqPan = 0.17
    else:
        raise NotImplementedError('satellite: QuickBird or IKONOS')
    # lowpass
    img_ = img.squeeze()
    img_ = img_.astype(np.float64)
    if img_.ndim == 2:  # Pan
        H, W = img_.shape
        lowpass = GNyq2win(GNyqPan, scale, N=41)
    elif img_.ndim == 3:  # MS
        H, W, _ = img.shape
        lowpass = [GNyq2win(gnyq, scale, N=41) for gnyq in GNyq]
        lowpass = np.stack(lowpass, axis=-1)
    img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
    # downsampling
    output_size = (H // scale, W // scale)
    img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)
    return img_








def _qindex(img1 ,img2 ,block_size):

    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1 ,'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size ,block_size)) / (block_size ** 2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size / 2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_ ,-1 ,window)[pad_topleft:-pad_bottomright ,pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_ ,-1 ,window)[pad_topleft:-pad_bottomright ,pad_topleft:-pad_bottomright]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1_ ** 2 ,-1 ,window)[pad_topleft:-pad_bottomright ,
                pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_ ** 2 ,-1 ,window)[pad_topleft:-pad_bottomright ,
                pad_topleft:-pad_bottomright] - mu2_sq
    #    print(mu1_mu2.shape)
    # print(sigma2_sq.shape)
    sigma12 = cv2.filter2D(img1_ * img2_ ,-1 ,window)[pad_topleft:-pad_bottomright ,
              pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0

    #    print(np.min(sigma1_sq + sigma2_sq), np.min(mu1_sq + mu2_sq))

    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = ((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
            (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])


    return np.mean(qindex_map)

def D_lambda(img_fake, img_lm, block_size=32, p=1):
    """Spectral distortion
    img_fake, generated HRMS
    img_lm, LRMS"""
    assert img_fake.ndim == img_lm.ndim == 3, 'Images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # D_lambda
    Q_fake = []
    Q_lm = []
    for i in range(C_f):
        for j in range(i+1, C_f):
            # for fake
            band1 = img_fake[..., i]
            band2 = img_fake[..., j]
            Q_fake.append(_qindex(band1, band2, block_size=block_size))
            # for real
            band1 = img_lm[..., i]
            band2 = img_lm[..., j]
            Q_lm.append(_qindex(band1, band2, block_size=block_size))
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    D_lambda_index = (np.abs(Q_fake - Q_lm) ** p).mean()


    return D_lambda_index ** (1/p)


def D_s(img_fake, img_lm, pan, satellite, scale=4, block_size=32, q=1):
    """Spatial distortion
    img_fake, generated HRMS
    img_lm, LRMS
    pan, HRPan"""
    # fake and lm
    assert img_fake.ndim == img_lm.ndim == 3, 'MS images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert H_f // H_r == W_f // W_r == scale, 'Spatial resolution should be compatible with scale'
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # fake and pan
    assert pan.ndim == 3, 'Panchromatic image must be 3D!'
    H_p, W_p, C_p = pan.shape
    assert C_p == 1, 'size of 3rd dim of Panchromatic image must be 1'
    assert H_f == H_p and W_f == W_p, "Pan's and fake's spatial resolution should be the same"
    # get LRPan, 2D
    pan_lr = mtf_resize(pan, satellite=satellite, scale=scale)
    #print(pan_lr.shape)
    # D_s
    Q_hr = []
    Q_lr = []
    for i in range(C_f):
        # for HR fake
        band1 = img_fake[..., i]
        band2 = pan[..., 0] # the input PAN is 3D with size=1 along 3rd dim
        #print(band1.shape)
        #print(band2.shape)
        Q_hr.append(_qindex(band1, band2, block_size=block_size))
        band1 = img_lm[..., i]
        band2 = pan_lr  # this is 2D
        #print(band1.shape)
        #print(band2.shape)
        Q_lr.append(_qindex(band1, band2, block_size=block_size))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    D_s_index = (np.abs(Q_hr - Q_lr) ** q).mean()

    return D_s_index ** (1/q)

def qnr(img_fake, img_lm, pan, satellite, scale=4, block_size=32, p=1, q=1, alpha=1, beta=1):
    """QNR - No reference IQA"""
    D_lambda_idx = D_lambda(img_fake, img_lm, block_size, p)
    D_s_idx = D_s(img_fake, img_lm, pan, satellite, scale, block_size, q)
    QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
    return D_lambda_idx,D_s_idx,QNR_idx



