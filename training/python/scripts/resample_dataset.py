import argparse
import os
import glob

from tqdm import tqdm
import imageio
import cv2
import time


CONST_PNG = cv2.IMWRITE_PNG_COMPRESSION


def resample_single_frame(frame_in_path, frame_out_path, height, width):
    # print("From: {} / To: {}".format(frame_in_path, frame_out_path))
    img = cv2.imread(frame_in_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(frame_out_path, img, (CONST_PNG, 0))


def resample(dataset_from, dataset_to, height, width):
    height = int(height)
    width = int(width)
    print(dataset_from)
    print(dataset_to)
    print(height)
    print(width)

    if not os.path.exists(dataset_from):
        raise FileNotFoundError("Dataset path does not exist: {}".format(dataset_from))

    if os.path.exists(dataset_to):
        raise ValueError("Directory already exists: {}".format(dataset_to))
    else:
        print("Creating directory: {}".format(dataset_to))
        os.makedirs(dataset_to)

    in_frames = glob.glob("{}/*/*/depth-*.png".format(dataset_from))
    teacher_frames = glob.glob("{}/*/*/_teacher.png".format(dataset_from))

    if len(in_frames) == 0 or len(teacher_frames) == 0:
        raise ValueError("No frames found in expected subdirectories. Are you sure the path points to a correct "
                         "dataset? Num frames: {}, num teacher frames: {}".format(len(in_frames), len(teacher_frames)))

    print("Found {} frames and {} teachers".format(len(in_frames), len(teacher_frames)))

    for frame_in_path in tqdm(in_frames + teacher_frames):

        # determine full path of new frame
        frame_out_path = frame_in_path.replace(dataset_from, dataset_to)

        # make sure the child directories always exist
        frame_out_dir = os.path.dirname(frame_out_path)
        if not os.path.exists(frame_out_dir):
            os.makedirs(frame_out_dir)

        resample_single_frame(frame_in_path, frame_out_path, height, width)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Resample dataset to a new target size")
    parser.add_argument("--dataset", help="Path to existing dataset directory")
    parser.add_argument("--target", help="Path to target dataset directory")
    parser.add_argument("--height", help="Target height")
    parser.add_argument("--width", help="Target width")

    args = parser.parse_args()
    resample(args.dataset, args.target, args.height, args.width)