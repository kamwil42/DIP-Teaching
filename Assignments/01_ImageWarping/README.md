# Assignment 1 - Image Warping

## Implementation of Image Geometric Transformation

This repository is Kamila Wilczyńska's implementation of Assignment_01 of DIP. 

<img src="pics/teaser1.png" alt="alt text" width="800">

## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```

## Running

To run basic transformation, run:

```basic
python run_global_transform.py
```

To run point guided transformation, run:

```point
python run_point_transform.py
```

## Results (need add more result images)
### Basic Transformation
<img src="pics/global_demo1.gif" alt="alt text" width="800">

### Point Guided Deformation:
<img src="pics/point_demo1.gif" alt="alt text" width="800">

#### Point Guided Deformation Points Removal:
<img src="pics/point_remove_demo.gif" alt="alt text" width="800">

#### Point Guided Deformation Incorrect Points Amount Warning:
<img src="pics/point_warning_demo.gif" alt="alt text" width="800">

## Acknowledgement

### Resources:
- [Teaching Slides](https://pan.ustc.edu.cn/share/index/66294554e01948acaf78) 
- [Paper: Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf)
- [OpenCV Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [Gradio: 一个好用的网页端交互GUI](https://www.gradio.app/)

>📋 Thanks for the algorithms proposed by [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).
