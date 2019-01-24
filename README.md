# Localizing salient body motion in multi-person scenes using convolutional neural networks

This repository contains neural network implementations and training code as used in our publication, along with code to create new instances of our dataset with differenct configurations.

## Publication

_Florian Letsch, Doreen Jirak, Stefan Wermter_\
**Localizing salient body motion in multi-person scenes using convolutional neural networks**

https://www.sciencedirect.com/science/article/pii/S0925231218313791

## What's included

- `training/`: Code to run network training. Has its own [README.md](training/README.md). Requires download of "v1" dataset package.
- `dataset/`: Code to stitch a dataset. Has its own [README.md](dataset/README.md). Requires download of "raw data" package. 

## Available downloads

- **V1 dataset package** (1.2 GB): Train, validation and test data to be used for learning the detection and localization task described in our publication. Download from https://florianletsch.de/media/publication/kinect-gestures-v1-240x320.zip - Afterwards you will want to look at the [README.md](training/README.md) inside the `training/` subfolder.
- **Raw data package** (10.7 GB): Kinect frames and annotations which can be used to create a _new_ instance of a dataset used for training. Download instructions can be found at https://www.inf.uni-hamburg.de/en/inst/ab/wtm/research/corpora.html - Afterwards you will want to look at the [README.md](dataset/README.md) inside the `dataset/` subfolder.

## Contact

For inquiries, please contact florian.letsch@20bn.com

## How to cite via bibtex

```tex
@article{LETSCH2019449,
title = "Localizing salient body motion in multi-person scenes using convolutional neural networks",
journal = "Neurocomputing",
volume = "330",
pages = "449 - 464",
year = "2019",
issn = "0925-2312",
doi = "https://doi.org/10.1016/j.neucom.2018.11.048",
url = "http://www.sciencedirect.com/science/article/pii/S0925231218313791",
author = "Florian Letsch and Doreen Jirak and Stefan Wermter",
keywords = "Convolutional neural networks, Gestures, Computer vision, Detection, Localization, Saliency",
abstract = "With modern computer vision techniques being successfully developed for a variety of tasks, extracting meaningful knowledge from complex scenes with multiple people still poses problems. Consequently, experiments with application-specific motion, such as gesture recognition scenarios, are often constrained to single person scenes in the literature. Therefore, in this paper we address the challenging task of detecting salient body motion in scenes with more than one person. We propose a neural architecture that only reacts to a specific kind of motion in the scene: A limited set of body gestures. The model is trained end-to-end, thereby avoiding hand-crafted features and the strong reliance on pre-processing as it is prevalent in similar studies. The presented model implements a saliency mechanism that reacts to body motion cues which have not been included in previous computational saliency systems. Our architecture consists of a 3D Convolutional Neural Network that receives a frame sequence as its input and localizes active gesture movement. To train our network with a large data variety, we introduce an approach to combine Kinect recordings of one person into artificial scenes with multiple people, yielding a large diversity of scene configurations in our dataset. We performed experiments using these sequences and show that the proposed model is able to localize the salient body motion of our gesture set. We found that 3D convolutions and a baseline model with 2D convolutions perform surprisingly similar on our task. Our experiments revealed the influence of gesture characteristics on how well they can be learned by our model. Given a distinct gesture set and computational restrictions, we conclude that using 2D convolutions might often perform equally well."
}
```
