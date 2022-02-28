# SRDenseNet-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Image Super-Resolution Using Dense Skip Connections](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf).

### Table of contents

- [SRDenseNet-PyTorch](#srdensenet-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Image Super-Resolution Using Dense Skip Connections](#about-image-super-resolution-using-dense-skip-connections)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
        - [Image Super-Resolution Using Dense Skip Connections](#image-super-resolution-using-dense-skip-connections)

## About Image Super-Resolution Using Dense Skip Connections

If you're new to SRDenseNet, here's an abstract straight from the paper:

Recent studies have shown that the performance of single-image super-resolution methods can be significantly boosted by using deep convolutional
neural networks. In this study, we present a novel single-image super-resolution method by introducing dense skip connections in a very deep network.
In the proposed network, the feature maps of each layer are propagated into all subsequent layers, providing an effective way to combine the low-level
features and high-level features to boost the reconstruction performance. In addition, the dense skip connections in the network enable short paths to
be built directly from the output to each layer, alleviating the vanishing-gradient problem of very deep networks. Moreover, deconvolution layers are
integrated into the network to learn the upsampling filters and to speedup the reconstruction process. Further, the proposed method substantially
reduces the number of parameters, enhancing the computational efficiency. We evaluate the proposed method using images from four benchmark datasets
and set a new state of the art.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Test

Modify the contents of the file as follows.

- line 30: `upscale_factor` change to the magnification you need to enlarge.
- line 32: `mode` change Set to valid mode.
- line 69: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 30: `upscale_factor` change to the magnification you need to enlarge.
- line 32: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

- line 47: `start_epoch` change number of training iterations in the previous round.
- line 48: `resume` the weight address that needs to be loaded.

## Result

Source of original paper results: https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       | 
|:-------:|:-----:|:----------------:|
|  Set5   |   4   | 32.02(**31.50**) |
|  Set14  |   4   | 28.50(**28.00**) |

Low Resolution / Super Resolution / High Resolution
<span align="center"><img src="assets/result.png"/></span>

### Credit

#### Image Super-Resolution Using Dense Skip Connections

_Tong, Tong and Li, Gen and Liu, Xiejie and Gao, Qinquan_ <br>

**Abstract** <br>
Recent studies have shown that the performance of single-image super-resolution methods can be significantly boosted by using deep convolutional
neural networks. In this study, we present a novel single-image super-resolution method by introducing dense skip connections in a very deep network.
In the proposed network, the feature maps of each layer are propagated into all subsequent layers, providing an effective way to combine the low-level
features and high-level features to boost the reconstruction performance. In addition, the dense skip connections in the network enable short paths to
be built directly from the output to each layer, alleviating the vanishing-gradient problem of very deep networks. Moreover, deconvolution layers are
integrated into the network to learn the upsampling filters and to speedup the reconstruction process. Further, the proposed method substantially
reduces the number of parameters, enhancing the computational efficiency. We evaluate the proposed method using images from four benchmark datasets
and set a new state of the art.

[[Paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)

```
@inproceedings{tong2017image,
  title={Image super-resolution using dense skip connections},
  author={Tong, Tong and Li, Gen and Liu, Xiejie and Gao, Qinquan},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={4799--4807},
  year={2017}
}
```
