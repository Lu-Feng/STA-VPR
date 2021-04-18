# STA-VPR
This repository provides the code for our paper "[STA-VPR: Spatio-temporal Alignment for Visual Place Recognition](https://ieeexplore.ieee.org/document/9382071)" by Feng Lu, Baifan Chen, Xiang-Dong Zhou, and Dezhen Song. The arXiv pre-print can found [here](https://arxiv.org/abs/2103.13580). The repository is still being updated.

![spatial_alignment_sample](images/spatial_alignment_sample.png)

## Installation
- Python ≥3.5
- cuda ≥9.0
- Numba ≥0.44.1
- Keras ≥2.2.4
- TensorFlow ≥1.10
- Pytorch ≥1.1

The Keras and TensorFlow are only required by the VGG16 model (will be downloaded when you run the code). You can just install PyTorch if you only need to use the DenseNet16 model.

## Steps
- Download the [DenseNet161](http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar) model pretrained on [Places365](https://github.com/CSAILVision/places365) and copy them to the STA-VPR folder. Note that if you use this pretrained model, please see the License required by [Places365](https://github.com/CSAILVision/places365).
- You will need to update the configuration file "STAVPRconfig.yaml" changing some information, such as the file pathes of your images, model name, and so on.
- Run the 'STAVPR_demo.py' (python3 STAVPR_demo.py)

## Acknowledgements 
Arren Glover for the Gardens Point dataset

## Citation

Please consider citing the following publication if you use any part of the provided code:
```
@ARTICLE{stavpr,
author={F. {Lu} and B. {Chen} and X. -D. {Zhou} and D. {Song}},
journal={IEEE Robotics and Automation Letters},
title={STA-VPR: Spatio-Temporal Alignment for Visual Place Recognition},
year={2021},
volume={6},
number={3},
pages={4297-4304},
doi={10.1109/LRA.2021.3067623}
}
```
