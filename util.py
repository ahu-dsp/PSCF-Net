import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import math
import numpy as np
import torch
from cross_correlation import xcorr_torch
from spectral_tools import gen_mtf
import torch.nn as nn
from math import floor
from skimage.transform import resize
from interpolator_tools import interp23tap

def save_figure(losses, path, epoch, label):


    # except:
    if len(losses) == 2:
        plt.plot(losses[0], label='adv-loss', color='r')
        plt.plot(losses[1], label='recon-loss', color='g')

    else:
        plt.plot(losses, label=label, color='r')
        plt.title("Experiment: {} -- {}: {}".format(path, label, epoch))

    plt.legend()
    plt.savefig("results-{}/epoch{}-{}-loss.pdf".format(path, epoch, label,))
    plt.close()

def scale_range(input, min, max):
    input += -(np.min(input))
    input /= (1e-9 + np.max(input) / (max - min + 1e-9))
    input += min
    return input

def rgb2gray(rgb):
    r, g, b, nir = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], rgb[:, :, 3]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = 0.25 * r + 0.25 * g + 0.25 * b + 0.25 * nir
    return gray

def visualize_tensor(imgs, epoch, it, name):
    fname = "tensors-{}/{}/{}-{}.jpg".format(opt.savePath, epoch, it, name)
    vutils.save_image(
        tensor=imgs, filename=fname, normalize=True, nrow=imgs.size()[0] // 2)

def avg_metric(target, prediction, metric):
    sum = 0
    batch_size = len(target)
    for i in range(batch_size):
        sum += metric(np.transpose(target.data.cpu().numpy()
                                   [i], (1, 2, 0)), np.transpose(prediction.data.cpu().numpy()[i], (1, 2, 0)))
    return sum/batch_size


def net_scope(kernel_size):
    """
        Compute the network scope.

        Parameters
        ----------
        kernel_size : List[int]
            A list containing the kernel size of each layer of the network.

        Return
        ------
        scope : int
            The scope of the network

        """

    scope = 0
    for i in range(len(kernel_size)):
        scope += math.floor(kernel_size[i] / 2)
    return scope


def local_corr_mask(img_in, ratio, sensor, device, kernel=8):
    """
        Compute the threshold mask for the structural loss.

        Parameters
        ----------
        img_in : Torch Tensor
            The test image, already normalized and with the MS part upsampled with ideal interpolator.
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        device : Torch device
            The device on which perform the operation.
        kernel : int
            The semi-width for local cross-correlation computation.
            (See the cross-correlation function for more details)

        Return
        ------
        mask : PyTorch Tensor
            Local correlation field stack, composed by each MS and PAN. Dimensions: Batch, B, H, W.

        """

    I_PAN = torch.unsqueeze(img_in[:, -1, :, :], dim=1)
    I_MS = img_in[:, :-1, :, :]

    MTF_kern = gen_mtf(ratio, sensor)[:, :, 0]
    MTF_kern = np.expand_dims(MTF_kern, axis=(0, 1))
    MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)
    pad = floor((MTF_kern.shape[-1] - 1) / 2)

    padding = nn.ReflectionPad2d(pad)

    depthconv = nn.Conv2d(in_channels=1,
                          out_channels=1,
                          groups=1,
                          kernel_size=MTF_kern.shape,
                          bias=False)

    depthconv.weight.data = MTF_kern
    depthconv.weight.requires_grad = False

    I_PAN = padding(I_PAN)
    I_PAN = depthconv(I_PAN)
    mask = xcorr_torch(I_PAN, I_MS, kernel, device)
    mask = 1.0 - mask

    return mask

def show(starting_img_ms, img_pan, algorithm_outcome, ratio, method, q_min=0.02, q_max=0.98):
    """
        Auxiliary function for results visualization.

        Parameters
        ----------
        starting_img_ms : Numpy Array
            The Multi-Spectral image. Dimensions: H, W, Bands
        img_pan : Numpy Array
            The PAN image. Dimensions: H, W
        algorithm_outcome : NumPy Array
            The Fused image. Dimensions: H, W, Bands
        ratio : int
            PAN-MS resolution ratio
        method : str
            The name of the pansharpening algorithm
        q_min : float
            Minimum quantile to compute, which must be between 0 and 1 inclusive.
        q_max : float
            Maximum quantile to compute, which must be between 0 and 1 inclusive.

        Return
        ------
        None

    """

    Q_MS = np.quantile(starting_img_ms, (q_min, q_max), (0, 1), keepdims=True)
    Q_PAN = np.quantile(img_pan, (q_min, q_max), (0, 1), keepdims=True)

    ms_shape = (starting_img_ms.shape[0] * ratio, starting_img_ms.shape[1] * ratio, starting_img_ms.shape[2])

    I_MS_LR_4x = resize(starting_img_ms, ms_shape, order=0)
    I_interp = interp23tap(starting_img_ms, ratio)

    DP = algorithm_outcome - I_interp
    Q_d = np.quantile(abs(DP), q_max, (0, 1))
    if starting_img_ms.shape[-1] == 8:
        RGB = (4, 2, 1)
        RYB = (4, 3, 1)
    else:
        RGB = (2, 1, 0)
        RYB = (2, 3, 0)
    plt.figure()
    ax1 = plt.subplot(2, 4, 1)
    plt.imshow((img_pan - Q_PAN[0, :, :]) / (Q_PAN[1, :, :] - Q_PAN[0, :, :]), cmap='gray')
    ax1.set_title('PAN')

    T = (I_MS_LR_4x - Q_MS[0, :, :]) / (Q_MS[1, :, :] - Q_MS[0, :, :])
    T = np.clip(T, 0, 1)

    ax2 = plt.subplot(2, 4, 2, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RGB])
    ax2.set_title('MS (RGB)')

    ax6 = plt.subplot(2, 4, 6, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RYB])
    ax6.set_title('MS (RYB)')

    T = (algorithm_outcome - Q_MS[0, :, :]) / (Q_MS[1, :, :] - Q_MS[0, :, :])
    T = np.clip(T, 0, 1)

    ax3 = plt.subplot(2, 4, 3, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RGB])
    ax3.set_title(method + ' (RGB)')

    ax7 = plt.subplot(2, 4, 7, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RYB])
    ax7.set_title(method + ' (RYB)')

    T = 0.5 + DP / (2 * Q_d)
    T = np.clip(T, 0, 1)

    ax4 = plt.subplot(2, 4, 4, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RGB])
    ax4.set_title('Detail (RGB)')

    ax8 = plt.subplot(2, 4, 8, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RYB])
    ax8.set_title('Detail (RYB)')
    plt.show()
    return