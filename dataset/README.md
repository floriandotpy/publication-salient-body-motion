Dataset: Kinect Gestures
===

Raw data, annotations and dataset generation code to accompany the publication:
 
Florian Letsch, Doreen Jirak, Stefan Wermter:
_Localizing salient body motion in multi-person scenes using convolutional neural networks_
 
Link: https://www.sciencedirect.com/science/article/pii/S0925231218313791

## Changelog

### 1.0.0 (24 January, 2019)

- Initial release

## What this includes

- Raw Kinect recordings 
- Script to combine raw Kinect recordings into scenes with multiple people

## What this does not include

- The stitched dataset used during experiments of the publication. To find that dataset, please head to https://www.inf.uni-hamburg.de/en/inst/ab/wtm/research/corpora.html

## Setup

The following instructions were tested using Ubuntu 18.04

1. Install anaconda for Python package management and environments, details see: https://conda.io/miniconda.html
2. Create environment `conda create --name stitching --file conda-requirements.txt`
3. Activate environment. `conda activate stitching`


## Check that videos can be found

First, let's check if all paths are set correctly and both the depth frames 
and the annotations can be found.

```sh
export PYTHONPATH=.. 
cd scripts/
python stitch.py --check --video_path=/path/to/videos
```

This should produce an output like the following.

- **Boxes** are the annotations of bounding boxes for all recordings that contain people
- **Times** are the annotations of gesture executions (when in the recording does one gesture start and stop)
- **Depth** are the depth frames that came along with each recording, but are saved separately so that they can be read during dataset generation (=stitching)


```
[check] Boxes: YES  Times: YES  Depth: YES  |  s1-abort-depth
[check] Boxes: YES  Times: YES  Depth: YES  |  s1-check-depth
[check] Boxes: YES  Times: YES  Depth: YES  |  s1-chi-depth
[check] Boxes: YES  Times: YES  Depth: YES  |  s1-circle-depth
...
```

Exactly four rows should contain `NO` for both `Boxes` and `Times` annotations. 
This is okay, as these were not
annotated and are not used during stitching. These 4 videos are:
 
```
[check] Boxes: NO   Times: NO   Depth: YES  |  s2-point-left-depth
...
[check] Boxes: NO   Times: NO   Depth: YES  |  s1-background-walking-depth
...
[check] Boxes: NO   Times: NO   Depth: YES  |  s1-walking-left-depth
[check] Boxes: NO   Times: NO   Depth: YES  |  s1-walking-right-depth
``` 
 
All other rows should say `YES` in all three columns though.
 
## Run stitching script to generate dataset

Assuming all data can be found, your can use the main script `stitch.py` to
start the dataset generation. Set parameters:
 
- `--video_path=/path/to/videos` : Path to the original depth frames
- `--out_path=/path/to/dataset`: A reasonable output directory (**will be overwritten by default**).
 
Then, you can start the dataset generation as follows. Stitching will take a while (in the range of around 2 hours, 
depending on your hardware).

```sh
export PYTHONPATH=.. 
cd scripts/
python stitch.py --video_path=/path/to/videos --out_path=/path/to/dataset
```

This will create a new dataset "instance" -- so a stitched version generated from the original 
recordings in the following structure (Total size will be about 3.2GB):

```
/path/to/dataset/
    fold_0/
        1536002281444_s4-standing-depth
        1536002282189_s4-standing-depth
        1536002283087_s2-wave-depth
        1536002283884_s2-wave-depth
        1536002284971_s2-wave-depth_s4-standing-depth
        ...
        
    fold_1/
        ...
        
    fold_2/
        ...
        
    fold_3/
        ...
        
    fold_4/
        ...
```