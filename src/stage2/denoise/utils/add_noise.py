# There are functions for creating a train and validation iterator.
import os
import random
import re
import threading
from functools import partial
from itertools import product
from os import mkdir

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.widgets import Slider
from PIL import Image
from scipy.io import loadmat, savemat
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose


class BaseNormalizer:
    def __init__(self):
        assert hasattr(self, "STATEFUL"), "Missing STATEFUL class attribute"

    def fit(self, x):
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    def get_id(self):
        attributes = [self.__class__.__name__]
        attributes += [k[:3] + str(v) for k, v in self.__dict__.items() if not isinstance(v, torch.Tensor)]
        return "_".join(attributes).replace(".", "")

    def __repr__(self):
        return self.get_id()

    def filename(self):
        return f"{self.get_id()}.pth"

    def save(self, path=None):
        filename = self.filename()
        if path:
            filename = os.path.join(path, filename)
        torch.save(self.__dict__, filename)

    def load(self, path=None):
        filename = self.filename()
        if path:
            filename = os.path.join(path, filename)
        state = torch.load(filename)
        for k, v in state.items():
            setattr(self, k, v)


class BandMinMaxQuantileStateful(BaseNormalizer):
    STATEFUL = True

    def __init__(self, low=0.02, up=0.98, epsilon=0.001):
        super().__init__()
        self.low = low
        self.up = up
        self.epsilon = epsilon

    def fit(self, imgs):
        x_train = []
        for i, img in enumerate(imgs):
            x_train.append(img.flatten(start_dim=1))
        x_train = torch.cat(x_train, dim=1)
        bands = x_train.shape[0]
        q_global = np.zeros((bands, 2))
        for b in range(bands):
            q_global[b] = np.percentile(x_train[b].cpu().numpy(), q=100 * np.array([self.low, self.up]))

        self.q = torch.tensor(q_global, dtype=torch.float32).T[..., None, None]

    def transform(self, x):
        x = torch.minimum(x, self.q[1])
        x = torch.maximum(x, self.q[0])
        return (x - self.q[0]) / (self.epsilon + (self.q[1] - self.q[0]))


def Data2Volume(data, ksizes, strides):
    """
    Construct Volumes from Original High Dimensional (D) Data
    """
    dshape = data.shape
    PatNum = lambda l, k, s: (np.floor((l - k) / s) + 1)

    TotalPatNum = 1
    for i in range(len(ksizes)):
        TotalPatNum = TotalPatNum * PatNum(dshape[i], ksizes[i], strides[i])

    V = np.zeros([int(TotalPatNum)] + ksizes)  # create D+1 dimension volume

    args = [range(kz) for kz in ksizes]
    for s in product(*args):
        s1 = (slice(None),) + s
        s2 = tuple([slice(key, -ksizes[i] + key + 1 or None, strides[i]) for i, key in enumerate(s)])
        V[s1] = np.reshape(data[s2], -1)

    return V


def crop_center(img, cropx, cropy):
    _, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty : starty + cropy, startx : startx + cropx]


def rand_crop(img, cropx, cropy):
    _, y, x = img.shape
    x1 = random.randint(0, x - cropx)
    y1 = random.randint(0, y - cropy)
    return img[:, y1 : y1 + cropy, x1 : x1 + cropx]


def sequetial_process(*fns):
    """
    Integerate all process functions
    """

    def processor(data):
        for f in fns:
            data = f(data)
        return data

    return processor


def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


def frame_diff(frames):
    diff_frames = frames[1:, ...] - frames[:-1, ...]
    return diff_frames


def visualize(filename, matkey, load=loadmat, preprocess=None):
    """
    Visualize a preprecessed hyperspectral image
    """
    if not preprocess:
        preprocess = lambda identity: identity
    mat = load(filename)
    data = preprocess(mat[matkey])
    print(data.shape)
    print(np.max(data), np.min(data))

    data = np.squeeze(data[:, :, :])
    Visualize3D(data)
    # Visualize3D(np.squeeze(data[:,0,:,:]))


def Visualize3D(data, meta=None):
    data = np.squeeze(data)

    for ch in range(data.shape[0]):
        data[ch, ...] = minmax_normalize(data[ch, ...])

    print(np.max(data), np.min(data))

    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    frame = 0
    # l = plt.imshow(data[frame,:,:])

    l = plt.imshow(data[frame, :, :], cmap="gray")  # shows 256x256 image, i.e. 0th frame
    # plt.colorbar()
    axcolor = "lightgoldenrodyellow"
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sframe = Slider(axframe, "Frame", 0, data.shape[0] - 1, valinit=0)

    def update(val):
        frame = int(np.around(sframe.val))
        l.set_data(data[frame, :, :])
        if meta is not None:
            axframe.set_title(meta[frame])

    sframe.on_changed(update)

    plt.show()


def data_augmentation(image, mode=None):
    """
    Args:
        image: np.ndarray, shape: C X H X W
    """
    axes = (-2, -1)
    flipud = lambda x: x[:, ::-1, :]

    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = flipud(image)

    # we apply spectrum reversal for training 3D CNN, e.g. QRNN3D.
    # disable it when training 2D CNN, e.g. MemNet
    if random.random() < 0.5:
        image = image[::-1, :, :]

    return np.ascontiguousarray(image)


class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()


class Augment_RGB_torch:
    def __init__(self):
        pass

    def transform0(self, torch_tensor):
        return torch_tensor

    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1, -2])
        return torch_tensor

    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1, -2])
        return torch_tensor

    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1, -2])
        return torch_tensor

    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor

    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1, -2])).flip(-2)
        return torch_tensor

    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1, -2])).flip(-2)
        return torch_tensor

    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1, -2])).flip(-2)
        return torch_tensor


# Define Transforms
class RandomGeometricTransform(object):
    def __call__(self, img):
        """
        Args:
            img (np.mdarray): Image to be geometric transformed.

        Returns:
            np.ndarray: Randomly geometric transformed image.
        """
        if random.random() < 0.25:
            return data_augmentation(img)
        return img


class RandomCrop(object):
    """For HSI (c x h x w)"""

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        img = rand_crop(img, self.crop_size, self.crop_size)
        return img


class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True:
            # print(i)
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out


class AddNoise(object):
    """add gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, sigma):
        self.sigma_ratio = sigma / 255.0

    def __call__(self, img):
        noise = np.random.randn(*img.shape) * self.sigma_ratio
        # print(img.sum(), noise.sum())
        return img + noise


class AddNoiseBlind(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""

    def __pos(self, n):
        i = 0
        while True:
            yield i
            i = (i + 1) % n

    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.0
        self.pos = LockedIterator(self.__pos(len(sigmas)))

    def __call__(self, img):
        sigma = self.sigmas[next(self.pos)]
        noise = np.random.randn(*img.shape) * sigma
        return img + noise, sigma


class AddNoiseBlindv1(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, min_sigma, max_sigma):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        sigma = np.random.uniform(self.min_sigma, self.max_sigma) / 255
        noise = np.random.randn(*img.shape) * sigma
        # print(img.shape)
        out = img + noise
        return out  # , sigma


class AddNoiseBlindv2(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, min_sigma, max_sigma):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        sigma = np.random.uniform(self.min_sigma, self.max_sigma) / 255
        noise = np.random.randn(*img.shape) * sigma
        # print(img.shape)
        out = img + noise
        return out  # , sigma


class AddNoiseNoniid(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.0

    def __call__(self, img):
        bwsigmas = np.reshape(
            self.sigmas[np.random.randint(0, len(self.sigmas), img.shape[0])],
            (-1, 1, 1),
        )
        noise = np.random.randn(*img.shape) * bwsigmas
        return img + noise


class AddNoiseMixed(object):
    """add mixed noise to the given numpy array (B,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_bands: list of number of band which is corrupted by each item in noise_bank"""

    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img):
        B, H, W = img.shape
        all_bands = np.random.permutation(range(B))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            bands = all_bands[pos : pos + num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img


class _AddNoiseImpulse(object):
    """add impulse noise to the given numpy array (B,H,W)"""

    def __init__(self, amounts, s_vs_p=0.5):
        self.amounts = np.array(amounts)
        self.s_vs_p = s_vs_p

    def __call__(self, img, bands):
        # bands = np.random.permutation(range(img.shape[0]))[:self.num_band]
        bwamounts = self.amounts[np.random.randint(0, len(self.amounts), len(bands))]
        for i, amount in zip(bands, bwamounts):
            self.add_noise(img[i, ...], amount=amount, salt_vs_pepper=self.s_vs_p)
        return img

    def add_noise(self, image, amount, salt_vs_pepper):
        # out = image.copy()
        out = image
        p = amount
        q = salt_vs_pepper
        flipped = np.random.choice([True, False], size=image.shape, p=[p, 1 - p])
        salted = np.random.choice([True, False], size=image.shape, p=[q, 1 - q])
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = 0
        return out


class _AddNoiseStripe(object):
    """add stripe noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_stripe = np.random.randint(np.floor(self.min_amount * W), np.floor(self.max_amount * W), len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            stripe = np.random.uniform(0, 1, size=(len(loc),)) * 0.5 - 0.25
            img[i, :, loc] -= np.reshape(stripe, (-1, 1))
        return img


class AddNoiseNoniid_v2(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, min_sigma, max_sigma):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        bwsigmas = np.reshape(
            (np.random.rand(img.shape[0]) * (self.max_sigma - self.min_sigma) + self.min_sigma),
            (-1, 1, 1),
        )
        noise = np.random.randn(*img.shape) * bwsigmas / 255
        return img + noise


class _AddNoiseDeadline(object):
    """add deadline noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_deadline = np.random.randint(np.ceil(self.min_amount * W), np.ceil(self.max_amount * W), len(bands))
        for i, n in zip(bands, num_deadline):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            img[i, :, loc] = 0
        return img


class _AddNoiseinpainting(object):
    """add deadline noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount, num):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.num = num

    def __call__(self, img):
        num = np.random.randint(0, self.num)
        num = self.num
        B, H, W = img.shape
        width_deadline = np.random.randint(np.ceil(self.min_amount * W), np.ceil(self.max_amount * W), num)
        loc_start = np.random.randint(0, W - np.ceil(self.max_amount * W) - 1, num)
        # loc_start = [10, 50, 80, 120, 150, 190]
        # width_deadline = [10, 15, 5, 10, 20, 5]
        for loc, width in zip(loc_start, width_deadline):
            # print(width)
            img[:, :, loc : loc + width] = 0
        return img


class AddNoiseImpulse(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])]
        self.num_bands = [1 / 3]


class AddNoiseStripe(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseStripe(0.05, 0.15)]
        self.num_bands = [1 / 3]


class AddNoiseDeadline(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseDeadline(0.05, 0.15)]
        self.num_bands = [1 / 3]


class AddNoiseInpainting(AddNoiseMixed):
    def __init__(self, num):
        self.noise_bank = _AddNoiseinpainting(0.00, 0.10, num)
        self.num_bands = [1.0]

    def __call__(self, img):
        B, H, W = img.shape
        img = self.noise_bank(img)
        return img


class AddNoiseComplex(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [
            _AddNoiseStripe(0.05, 0.15),
            _AddNoiseDeadline(0.05, 0.15),
            _AddNoiseImpulse([0.1, 0.3, 0.5, 0.7]),
        ]
        self.num_bands = [1 / 3, 1 / 3, 1 / 3]


class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """

    def __init__(self, use_2dconv):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            img = torch.from_numpy(hsi[None])
        # for ch in range(hsi.shape[0]):
        #     hsi[ch, ...] = minmax_normalize(hsi[ch, ...])
        # img = torch.from_numpy(hsi)

        return img.float()


def addNoiseGaussian(srcdir, dstdir):
    s_sigma = [10, 30, 50, 70]
    # s_sigma = [0]
    for sigma in s_sigma:
        dstdir_noise = dstdir + "/512_" + str(sigma)
        if not os.path.exists(dstdir_noise):
            mkdir(dstdir_noise)
        noisemodel = AddNoise(sigma)
        c = 0
        for filename in os.listdir(srcdir):
            c = c + 1
            print(c)
            filepath = os.path.join(srcdir, filename)
            mat = loadmat(filepath)
            srchsi = mat["data"].transpose(2, 0, 1)
            noisyhsi = noisemodel(srchsi)
            n_sigma = sigma / 255

            savemat(
                os.path.join(dstdir_noise, filename),
                {
                    "gt": srchsi.transpose(1, 2, 0),
                    "sigma": n_sigma,
                    "input": noisyhsi.transpose(1, 2, 0),
                },
            )


def addNoiseComplex(srcdir, dstdir):
    sigmas = [30, 50, 70, 90]
    noise_models = []
    names = ["noniid", "impulse", "deadline", "stripe", "mixture"]
    noise_models.append(AddNoiseNoniid(sigmas))
    noise_models.append(AddNoiseImpulse())
    noise_models.append(AddNoiseDeadline())
    noise_models.append(AddNoiseStripe())
    add_noniid_noise = Compose(
        [
            AddNoiseNoniid(sigmas),
            AddNoiseComplex(),
        ]
    )
    noise_models.append(add_noniid_noise)

    for noise_name, noise_model in zip(names, noise_models):
        dstdir_noise = dstdir + "/512_" + noise_name
        c = 0
        if not os.path.exists(dstdir_noise):
            mkdir(dstdir_noise)
        for filename in os.listdir(srcdir):
            c = c + 1
            print(c)
            filepath = os.path.join(srcdir, filename)
            mat = loadmat(filepath)
            # srchsi = mat['data'].transpose(2,0,1)
            srchsi = mat["data"]

            noisyhsi = noise_model(srchsi)
            print(dstdir_noise, filepath)
            # savemat(os.path.join(dstdir_noise, filename), {'gt': srchsi.transpose(
            #    1, 2, 0), 'input': noisyhsi.transpose(1, 2, 0)})
            savemat(
                os.path.join(dstdir_noise, filename),
                {
                    "gt": loadmat(os.path.join(srcdir, filename))["data"],
                    "input": noisyhsi,
                },
            )


def addNoiseInpainting(srcdir, dstdir):
    dstdir_noise = dstdir + "/512_inpainting"
    if not os.path.exists(dstdir_noise):
        mkdir(dstdir_noise)
    noisemodel = AddNoiseInpainting(4)
    c = 0
    for filename in os.listdir(srcdir):
        c = c + 1
        print(c)
        filepath = os.path.join(srcdir, filename)
        mat = loadmat(filepath)
        srchsi = mat["data"]  # .transpose(2,0,1) # 191*200*200
        noisyhsi = noisemodel(srchsi)
        srchsi = loadmat(filepath)["data"]

        savemat(
            os.path.join(dstdir_noise, filename),
            {"gt": srchsi.transpose(1, 2, 0), "input": noisyhsi.transpose(1, 2, 0)},
        )


if __name__ == "__main__":
    # srcdir = "/mnt/code/users/yuchunmiao/hypersigma-master/data/Hyperspectral_Project/WDC/test"  # file path of original file
    # dstdir = "/mnt/code/users/yuchunmiao/hypersigma-master/data/Hyperspectral_Project/WDC/test_noise/complex"  # file path to put the testing file
    # # addNoiseComplex(srcdir,dstdir)
    # # addNoiseGaussian(srcdir,dstdir)
    # addNoiseInpainting(srcdir, dstdir)

    noise_complex = AddNoiseComplex()
    x = np.random.randn(32, 256, 256)
    y = noise_complex(x)
    print(y.shape)
    pass
