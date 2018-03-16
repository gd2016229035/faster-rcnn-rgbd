# A Simple and Fast Implementation of RGBD Faster R-CNN

## 1. Introduction

This project is a RGBD Faster R-CNN implementation based on [Chen Yun's faster rcnn](https://github.com/chenyuntc/simple-faster-rcnn-pytorch) . It aims to:

- Easily transform the origin RGB Faster R-CNN to RGBD version which can easily run on NYUV2 dataset
- Extract the feature from RGB and Depth images respectively and concat them at Conv5 layer followed by a NIN layer 
- Be straightly transfer to other muti-modality dataset(such as RGB-infrared)


## 2. Performance

Evaluate on [NYU Depth V2 dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
VGG16 train on `trainval` and test on `test` split. 

**Note**: You can run code in the `$faster-rcnn-RGB/` folder to get the RGB or Depth result(As the first two rows in the following table) by simplythe change one line `#127` in `$data/voc_dataset.py` file.Training shows great randomness, you may need a bit of luck and more epoches of training to reach the highest mAP. However, it should be easy to surpass the lower bound.

|              Implementation              |     mAP     |
| :--------------------------------------: | :---------: |
| RGB(~4G) |    0.337    |
| Depth(~4G)     | 0.348 |
| RGBD(concat at Conv5)(~5G)  | 0.394 |
| RGBD(concat at fc7)(~6G) |   0.386    |

## 3. Install dependencies

requires python3 and PyTorch 0.3

- install PyTorch >=0.3 with GPU (code are GPU-only), refer to [official website](http://pytorch.org)

- install cupy, you can install via `pip install` but it's better to read the [docs](https://docs-cupy.chainer.org/en/latest/install.html#install-cupy-with-cudnn-and-nccl) and make sure the environ is correctly set

- install other dependencies:  `pip install -r requirements.txt `

- Optional, but strongly recommended: build cython code `nms_gpu_post`: 

  ```Bash
  cd model/utils/nms/
  python3 build.py build_ext --inplace
  ```

- start vidom for visualization

```Bash
python3 -m visdom.server
```


## 5. Train

### 5.1 Prepare data


#### NYU Depth V2 dataset

1. Download the training, validation, test data from [Gupta 's dataset'](http://www.cs.berkeley.edu/~sgupta/eccv14/eccv14-data.tgz) or [the official website](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)

2. Transform the format of origin dataset to the format of "Pascal VOC2007".You can refer to [Gupta's code](https://github.com/s-gupta/nyu-hooks)

3. It should have this basic structure of 'nyuv2' folder

   ```Bash
   $Annotations/                 # annotations
   $ImageSets/                   # image list
   $JPEGImages/                  # RGB image sets
   $JPEGImages_depth/      # Depth image sets
   # ... and several other directories ...
   ```

4. modify `voc_data_dir` cfg item in `utils/config.py`, or pass it to program using argument like `--voc-data-dir=/path/to/nyuv2/` .


### 5 begin training

```Bash
mkdir checkpoints/ # folder for snapshots
```

```bash
sh train.sh    
```

you may refer to `utils/config.py` for more argument.

Some Key arguments:

- `--plot-every=n`: visualize prediction, loss etc every `n` batches.
- `--env`: visdom env for visualization
- `--voc_data_dir`: where the VOC data stored
- `--use-drop`: use dropout in RoI head, default False
- `--use-Adam`: use Adam instead of SGD, default SGD. (You need set a very low `lr` for Adam)
- `--load-path`: pretrained model path, default `None`, if it's specified, it would be loaded.

you may open browser, visit `http://<ip>:8097` and see the visualization of training procedure as below:

![visdom](http://7zh43r.com2.z0.glb.clouddn.com/del/visdom-fasterrcnn.png) 

If you're in China and encounter problem with visdom (i.e. timeout, blank screen), you may refer to [visdom issue](https://github.com/facebookresearch/visdom/issues/111#issuecomment-321743890), and see [troubleshooting](#troubleshooting) for solution.

## Troubleshooting
- visdom

  Some js files in visdom was blocked in China, see simple solution [here](https://github.com/chenyuntc/PyTorch-book/blob/master/README.md#visdom打不开及其解决方案)

  Also, `updata=append` doesn't work due to a bug brought in latest version, see [issue](https://github.com/facebookresearch/visdom/issues/233) and [fix](https://github.com/facebookresearch/visdom/pull/234/files)

  You don't need to build from source, modifying related files would be OK.

- dataloader: `received 0 items of ancdata` 

  see [discussion](https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667), It's alreadly fixed in [train.py](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/train.py#L17-L22). So I think you are free from this problem.
  
- cupy `numpy.core._internal.AxisError: axis 1 out of bounds [0, 1)`

  bug of cupy, see [issue](https://github.com/cupy/cupy/issues/793), fix via [pull request](https://github.com/cupy/cupy/pull/749)

  You don't need to build from source, modifying related files would be OK.

- VGG: Slow in construction

  VGG16 is slow in construction(i.e. 9 seconds),it could be speed up by this [PR](https://github.com/pytorch/vision/pull/377)
  
  You don't need to build from source, modifying related files would be OK.

- About the speed

  One strange thing is that, even the code doesn't use chainer, but if I remove `from chainer import cuda`, the speed drops a lot (train 6.5->6.1,test 14.5->10), because Chainer replaces the default allocator of CuPy by its memory pool implementation. But ever since V4.0, cupy use memory pool as default. However you need to build from souce if you are gona use the latest version of cupy (uninstall cupy -> git clone -> git checkout v4.0 -> setup.py install) @_@

  Another simple fix: add `from chainer import cuda` at the begining of `train.py`. in such case,you'll need to `pip install chainer` first.


## Acknowledgement
This work builds on many excellent works, which include:

- [Chen Yun's simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch) (mainly)

Licensed under MIT, see the LICENSE for more detail.

Contribution Welcome.

If you encounter any problem, feel free to open an issue.

Correct me if anything is wrong or unclear.

