# Assignment 2 - DIP with PyTorch

This repository is Kamila Wilczyńska's implementation of Assignment_02 of DIP.

## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```

## Running

To run Pix2Pix training, enter Pix2Pix folder and run:
```bash
bash download_facades_dataset.sh
python train.py
```
The provided code will train the model on the [Facades Dataset](https://cmp.felk.cvut.cz/~tylecr1/facade/).

## Results

### [Pix2Pix](https://phillipi.github.io/pix2pix/) with [Fully Convolutional Layers](https://arxiv.org/abs/1411.4038)

Dataset: [Facades Dataset](https://cmp.felk.cvut.cz/~tylecr1/facade/)

Loss Function: L1 Loss

Batch Size: 100

Epoch: 501

Optimizer: Adam with initial learning rate 0.001

Scheduler: StepLR with step size 200 and gamma 0.2

For result images, go to Assignment 2 home directory.

### Resources:
- [Assignment Slides](https://pan.ustc.edu.cn/share/index/66294554e01948acaf78)  
- [Paper: Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- Dataset citation:
@INPROCEEDINGS{Tylecek13,
  author = {Radim Tyle{\v c}ek and Radim {\v S}{\' a}ra},
  title = {Spatial Pattern Templates for Recognition of Objects with Regular Structure},
  booktitle = {Proc. GCPR},
  year = {2013},
  address = {Saarbrucken, Germany},
}
