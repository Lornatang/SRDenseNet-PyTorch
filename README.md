# SRDenseNet-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Image Super-Resolution Using Dense Skip Connections](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf).

### Table of contents

- [SRDenseNet-PyTorch](#srdensenet-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [Test](#test)
    - [Train SRDenseNet model](#train-srdensenet-model)
    - [Resume train SRDenseNet model](#resume-train-srdensenet-model)
    - [Result](#result)
    - [Credit](#credit)
        - [Image Super-Resolution Using Dense Skip Connections](#image-super-resolution-using-dense-skip-connections)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file. 

## Test

Modify the `config.py` file.

- line 29: `arch_name` change to `srdensenet_x4`.
- line 33: `upscale_factor` change to `4`.
- line 35: `mode` change to `test`.
- line 37: `exp_name` change to `test_SRDenseNet_x4`.
- line 82: `model_weights_path` change to `./results/pretrained_models/SRDenseNet_x4-ImageNet-bb28c23d.pth.tar`.

```bash
python3 test.py
```

## Train SRDenseNet model

Modify the `config.py` file.

- line 29: `arch_name` change to `srdensenet_x4`.
- line 33: `upscale_factor` change to `4`.
- line 35: `mode` change to `train`.
- line 37: `exp_name` change to `SRDenseNet_x4`.

```bash
python3 train.py
```

## Resume train SRDenseNet model

Modify the `config.py` file.

- line 29: `arch_name` change to `srdensenet_x4`.
- line 33: `upscale_factor` change to `4`.
- line 35: `mode` change to `train`.
- line 37: `exp_name` change to `SRDenseNet_x4`.
- line 54: `resume_model_weights_path` change to `./samples/SRDenseNet_x4/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       | 
|:-------:|:-----:|:----------------:|
|  Set5   |   4   | 32.02(**31.71**) |
|  Set14  |   4   | 28.50(**28.34**) |

```bash
# Download `SRGAN_x4-ImageNet-c71a4860.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py
```

Input: 

<span align="center"><img width="240" height="360" src="figure/comic_lr.png"/></span>

Output: 

<span align="center"><img width="240" height="360" src="figure/comic_sr.png"/></span>

```text
Build `srdensenet_x4` model successfully.
Load `srdensenet_x4` model weights `./results/pretrained_models/SRDenseNet_x4-ImageNet-bb28c23d.pth.tar` successfully.
SR image save to `./figure/comic_sr.png`
```

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
