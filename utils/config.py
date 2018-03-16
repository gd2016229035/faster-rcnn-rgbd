from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = '/usr/data/NYUD2/eccv14-data/data/nyuv2/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 4
    test_num_workers = 4

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3


    # visualization
    env = 'faster-rcnn-nyuv2'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'
<<<<<<< HEAD

=======
    concat_layer = 1
>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862
    # training
    epoch = 13


    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None
<<<<<<< HEAD
=======
    #load_rgb_path = '../faster-rcnn-model/rgb_34.71_vgg.pth'
    #load_depth_path = '../faster-rcnn-model/depth_35.77_vgg.pth'
    load_rgb_path = None
    load_depth_path = None

>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = './vgg16_caffe.pth'

<<<<<<< HEAD
=======
    load_depth_extimater='../depth_etimation/model_best.pth_L2_U-net_vgg11.tar'
    #load_rpn = './depth_35.77_rpn.pth'
    #load_head = './depth_35.77_head.pth'
    load_rpn = None
    load_head = None


>>>>>>> b43e1a358b5853ffb749ac931c9cd97a6dccf862
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
