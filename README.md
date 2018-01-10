# K-Means Clustering

----------

## Overview

The program partitions and clusters the pixel intentsities of a RGB image. Given any image size of *LxB* pixels and each having there components Red, Green and Blue.

Here we define the number of clusters of colors we want.

We will use *scikit-learn implementation of K-Means*, *matplotlib* to display out images and most dominant colors, we will use *argparse* to parse the line arguments, and finallt *cv2* to load images and do operations.

----------

## Setup

This project has the following dependencies:

- Python 3.5 or more
- IPython Notebook
- Scikit-learn
- Numpy
- OpenCV
- Matplotlib
- Argparse

----------

## Run a K-Means Clustering of an Image

```
$ python color_kmeans.py --image example.jpg --clusters 10
```
## Run Jupyter Notebook

```
$ jupyter notebook color_kmeans.ipynb
```