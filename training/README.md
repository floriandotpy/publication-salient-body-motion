Salient body motion: Training
===

In here you find the neural network models and training code as used in our publication.

The following instructions were tested using Ubuntu 18.04

## Preparation and dataset download

Create dataset and checkpoint directories and link them globally. If you prefer you can put the somewhere else than in your home folder, just make sure that the global `/datasets` and `/checkpoints` links are set up.

```
mkdir ~/checkpoints && mkdir ~/datasets
sudo ln -s ~/datasets /datasets
sudo ln -s ~/checkpoints /checkpoints
```

Download the dataset and extract it inside `/datasets`.

```
cd /datasets
wget https://florianletsch.de/media/publication/kinect-gestures-v1-240x320.zip
unzip kinect-gestures-v1-240x320.zip
rm kinect-gestures-v1-240x320.zip
```

## Environment setup

1. Install anaconda for Python package management and environments, details see: https://conda.io/miniconda.html
2. Create environment `conda create --name kinect-gestures --file conda-requirements.txt`
3. Activate environment. `conda activate kinect-gestures`

## Usage: Training

Make sure to activate your env!

```sh
conda activate kinect-gestures
```

Set the pythonpath to the package root directory and then run the training script with one of the provided example configs:

```sh
export PYTHONPATH=python/
python python/scripts/train.py configs/cnn3d.json
```

By default, training runs for 50 epochs. You may edit the config file to change this. On a GTX 1060, one epoch takes about 60 to 70 seconds, so a full training of the 3D CNN takes about 1 hour. The 2D CNN is smaller and training is faster (about 5x). VGG16 training is about 2x faster compared to the 3D CNN.

Training produces a line of output every epoch to monitor the process:

```
[...]
Epoch 2/50
 - 61s - loss: 0.1308 - motion_metric: 0.6131 - val_loss: 0.1059 - val_motion_metric: 0.6725
Epoch 3/50
 - 70s - loss: 0.1118 - motion_metric: 0.6677 - val_loss: 0.0922 - val_motion_metric: 0.7326
Epoch 4/50
 - 66s - loss: 0.1015 - motion_metric: 0.6999 - val_loss: 0.0837 - val_motion_metric: 0.7578
[...]
```

When training has finished, you will find a new directory inside `/checkpoints/cnn3d`.

It contains the following files:

- `config.json`: A copy of the config during trained, used to always look up original training parameters
- `history.json`: Different metrics tracked during training
- `metrics_test.json`: Metrics evaluated on the test set
- `out/`: Frames of the test set to visualize the final modell output
- `plot_loss.png`: Plot of training loss during training
- `plot_motion.png`: Plot of motion metric during training
- `weights.hdf5`: Model checkpoint


## Usage: Checkpoint evaluation

If you have an existing model checkpoint, you can re-evaluate metrics using this script.

```sh
export PYTHONPATH=python/
python python/scripts/eval_checkpoint.py /checkpoints/cnn3d
```

## Usage: Other included scripts

There is a handful of other more or less useful scripts (look at `python/scripts/`) included that you may want to look at if you decide to re-use our code base. ost of these have hard-coded paths to checkpoint or dataset instances, so you may need to edit a script before running it. 

- `determine_best.py`: Small utility: Opens a whole bunch of checkpoints and simply determines the "best" one.
- `plot_checkpoint.py`: Create plots from an existing checkpoint directory
- `produce_final_predictions.py`: Feed the test set through a bunch of checkpoints
- `resample_dataset.py`: Resample/rescale dataset. For faster local testing, you may want to downscale the dataset to e.g. 120x160 px.
- `run_experiments.py`: Auto generates a whole batch of experiment configs and runs them in sequence. Good for comparing a lot of model configs, or different models. 

## Usage: Run unit tests

There is (partial) unit test coverage for the training code base. Run as follows:

```sh
export PYTHONPATH=python/
nosetests python/tests/
```

This should output some Tensorflow debug logs and then finish with something like:

```
Ran 22 tests in 3.082s
OK
```

## Optional: Setup Slack notifications

If you want to keep using this training code as a base for your own experiments, it might be useful to setup Slack notifications. When running multiple experiments, you will get notifications whenever all experiments are completed, or whenever an exception has caused the training to stop early. Setup is easy, just have a look a `python/kinectgestures/notify.py` for instructions.
