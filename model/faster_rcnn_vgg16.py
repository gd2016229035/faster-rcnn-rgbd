# -*- coding: utf-8 -*-
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.models import alexnet
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt
<<<<<<< HEAD

=======
from unet_models import unet11
>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862

def decom_vgg16_classifier():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
<<<<<<< HEAD
=======
            print(' => load caffe_pretrain_path_classifier')
>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862
    else:
        model = vgg16(not opt.load_path)
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    return classifier


def decom_vgg16():
<<<<<<< HEAD
    # the 30th layer of features is relu of conv5_3
=======
>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
<<<<<<< HEAD
    else:
        model = vgg16(not opt.load_path)

    features = list(model.features)[:30]


    return nn.Sequential(*features)
=======
            print(' => load caffe_pretrain_path_rgb')
    else:
        model = vgg16(not opt.load_path)
    #5th conv1_2   64   
    #10th conv2_2   128
    #17th conv3_3   256
    #24th conv4_3   512
    #30th conv5_3   512
    features = list(model.features)[:30]
    features_net = nn.Sequential(*features)

    #features_others = list(model.features)[24:]
    #features_net_others = nn.Sequential(*features_others)
    if opt.load_rgb_path:
        features_net.load_state_dict(t.load(opt.load_rgb_path))
        print('==> load pretrained rgb model from %s' % opt.load_rgb_path)

    return features_net
>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862

def decom_vgg16_depth():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model2 = vgg16(pretrained=False)
        if not opt.load_path:
            model2.load_state_dict(t.load(opt.caffe_pretrain_path))
<<<<<<< HEAD
=======
            print(' => load caffe_pretrain_path_depth')
>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862
    else:
        model2 = vgg16(not opt.load_path)

    features2 = list(model2.features)[:30]
<<<<<<< HEAD

    # freeze top4 conv
    # for layer in features[:10]:
    #     for p in layer.parameters():
    #         p.requires_grad = False

    return nn.Sequential(*features2)


=======
    features2_net = nn.Sequential(*features2)

    if opt.load_depth_path:
        features2_net.load_state_dict(t.load(opt.load_depth_path))
        print('==> load pretrained depth model from %s' % opt.load_depth_path)




    return features2_net

model = unet11(pretrained='vgg')

>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862

class decom_vgg16_2stream(nn.Module):
    def __init__(self):
        # n_class includes the background
        super(decom_vgg16_2stream, self).__init__()
<<<<<<< HEAD
        self.extractor  = decom_vgg16()
        self.extractor2 = decom_vgg16_depth()
        self.NIN = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x,x2):
        x1_cnn1 = self.extractor(x)
        x2_cnn2 = self.extractor2(x2)       
        x_concat=t.cat((x1_cnn1,x2_cnn2),1)
        feature=self.NIN(x_concat)
        return feature
=======
        self.downpooling = nn.FractionalMaxPool2d(1,output_size=(256,320))
        self.upsample=nn.Upsample(size=[600,791], mode='bilinear')
        self.depth_extimater = model
        if opt.load_depth_extimater:
            self.depth_extimater.load_state_dict(t.load(opt.load_depth_extimater)['state_dict'])
            print(' ==> load depth_extimater successfully')
        self.extractor  = decom_vgg16()
        self.extractor2 = decom_vgg16_depth()
        self.NIN = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        #self.extractor_others = features_net_others
    def forward(self, x):
        x_small = self.downpooling(x)
        mydepth = self.depth_extimater(x_small)
        mydepth = self.upsample(mydepth)

        x1_cnn1 = self.extractor(x)
        mydepth_3channel = mydepth.repeat(1, 3, 1, 1)
        x2_cnn2 = self.extractor2(mydepth_3channel)
        x_concat = t.cat((x1_cnn1,x2_cnn2),1)
        feature = self.NIN(x_concat)
        #feature = self.features_net_others(feature)
        return feature,mydepth
>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862

class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=19,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor= decom_vgg16_2stream()
<<<<<<< HEAD
=======
        print(extractor)
>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862
        classifier=decom_vgg16_classifier()
        #extractor2 = decom_vgg16_depth()#### Depth

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = t.autograd.Variable(xy_indices_and_rois.contiguous())

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
