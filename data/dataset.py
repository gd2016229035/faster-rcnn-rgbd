import torch as t
from .voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from . import util
import numpy as np
from utils.config import opt


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        #img = img + (np.array([10.4898, 10.4898, 10.4898]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.24 + 0.4).clip(min=0, max=1) * 255

def inverse_normalize_depth(img):
    if opt.caffe_pretrain:
        img = img + (np.array([71.7645, 71.7645, 71.7645]).reshape(3, 1, 1))
        #img = img + (np.array([10.4898, 10.4898, 10.4898]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.099 + 0.28).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.4, 0.4, 0.4],
                                std=[0.24, 0.24, 0.24])
    img = normalize(t.from_numpy(img))
    return img.numpy()

def pytorch_normalze_depth(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
<<<<<<< HEAD
    normalize = tvtsf.Normalize(mean=[0.4, 0.4, 0.4],
                                std=[0.24, 0.24, 0.24])
    #normalize = tvtsf.Normalize(mean=[0.28, 0.28, 0.28],
    #                            std=[0.099, 0.099, 0.099])
=======
    normalize = tvtsf.Normalize(mean=[0.28, 0.28, 0.28],
                                std=[0.099, 0.099, 0.099])
>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862
    img = normalize(t.from_numpy(img))
    return img.numpy()


<<<<<<< HEAD
=======

>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862
def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    #mean = np.array([10.4898, 10.4898, 10.4898]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img

def caffe_normalize_depth(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([71.7645, 71.7645, 71.7645]).reshape(3, 1, 1)
    #mean = np.array([10.4898, 10.4898, 10.4898]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, img_depth, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
<<<<<<< HEAD
         (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray:
        A preprocessed image.
=======

    Returns:
        ~numpy.ndarray: A preprocessed image.
>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img_depth = img_depth / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
    img_depth = sktsf.resize(img_depth, (C, H * scale, W * scale), mode='reflect')
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
        normalize_depth = caffe_normalize_depth
    else:
        normalize = pytorch_normalze
        normalize_depth = pytorch_normalze_depth
    return normalize(img), normalize_depth(img_depth)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, img_depth, bbox, label = in_data
        _, H, W = img.shape
        img , img_depth = preprocess(img, img_depth ,self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, img_depth, params = util.random_flip(
            img, img_depth, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, img_depth, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, ori_img_depth, bbox, label, difficult = self.db.get_example(idx)

        img, img_depth, bbox, label, scale = self.tsf((ori_img, ori_img_depth, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), img_depth.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, ori_img_depth, bbox, label, difficult = self.db.get_example(idx)
        img, img_depth  = preprocess(ori_img, ori_img_depth)
        return img, img_depth, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
