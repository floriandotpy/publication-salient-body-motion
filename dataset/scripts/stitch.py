#!/usr/bin/python3

"""
A tool to stitch together several parts of annotated video files.
The result is a new frame sequence with an artificial scene that
can be used for training a classifier.
"""

from __future__ import print_function  # so that print(a, b) isnt interpreted as printing a tuple in Python 2
from __future__ import division  # so that 5 / 2 == 2.5 in both Python 2 and 3

import argparse
import sys
import os

# inject python source into PYTHONPATH
sys.path.insert(0, os.path.abspath('..'))

from stitcher.stitching import Stitch, SequenceGenerator, Scrambler, assert_verify_folds
from stitcher.recordings import Recordings

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Stitch together multiple videos on a background video from depth frames according to given annotations.')

    parser.add_argument('--check',
                        dest='check',
                        help="Check registered videos for available annotations",
                        action='store_true')
    parser.set_defaults(check=False)

    parser.add_argument('--test',
                        dest='test',
                        help='Generate sequences from annotated recordings for test phase',
                        action='store_true')
    parser.set_defaults(test=False)

    parser.add_argument('--video_path', default='/home/flo/datasets/UPLOAD/videos',
                        help="Path to where the original depth frames are stored.")
    parser.add_argument('--out_path', default='/home/flo/datasets/jobs-dev',
                        help="Path where to save the stitched dataset with generated sequences. Will be overwritten if it exists.")
    parser.add_argument('--fold_count', default=5, help="How many folds to split the dataset into.")

    args = parser.parse_args()

    _args = {
        'mode': Stitch.MODE_MASK,
        'destination': args.out_path,  # where to store all generated sequences. is created if not existing
        'scale_frames': 0.5,  # scale frames by this factor
        'scale_teacher': 0.5,  # scales teacher frames down by this factor
        'bg_strategy': 'random'  # background generation strategy
    }

    if args.check:

        Recordings(args.video_path).check()

    elif args.test:

        # test recordings are not used for generating "artificial sequences". for these, we only want to generate 15
        # frames long clips and the according teacher signal
        labeled_test = ['%s/%s' % (args.video_path, slug) for slug in
                        ['test/2-people-safe-img.avi']]

        for rec in labeled_test:
            generator = SequenceGenerator(rec, _args['scale_frames'], _args['scale_teacher'])
            generator.run('jobs-testing')

    else:

        '''
          What should a job description look like?
          - Specify a background file (probably no time labels needed, but might be nice to get variation in data)
          - Specify a list of actors, each with the following info
              - video file.
              - is performed gesture considered active?
              - where to place top/left corner of this actor
          Not defined, and determined automatically:
          - the video file of the actor contains multiple occurences of the gesture. all combinations of executed gestures
              should be rendered
          - the location of the saved job folder (frames + teacher signal) is generated automatically
          '''

        recordings = Recordings(args.video_path)

        labeled = recordings.get_labeled(include_active=True, include_passive=False)
        labeled_passive = recordings.get_labeled(include_active=False, include_passive=True)

        # cleanly split recorded videos in folds, so that cross validation without data overlap is possible
        count_folds = args.fold_count
        count_sequences_per_fold = 512
        count_scenes_per_fold = 55
        labeled_mix = Scrambler(labeled)
        labeled_passive_mix = Scrambler(labeled_passive)

        for fold_id in range(args.fold_count):
            count_fold_total = 0
            for i in range(count_scenes_per_fold):

                # empty scene
                scene_1 = []
                count_fold_total += Stitch(scene_1, args=_args, video_path=args.video_path,
                                           fold_count=args.fold_count).generate_jobs(
                    fold_id=fold_id,
                    count=1)  # does not get larger without refactoring

                # 1 person, passive
                scene_2 = [{'path': labeled_passive_mix.next(), 'active': False, 'fold_filter': fold_id}]
                count_fold_total += Stitch(scene_2, args=_args, video_path=args.video_path,
                                           fold_count=args.fold_count).generate_jobs(
                    fold_id=fold_id,
                    count=2)

                # 1 person, active
                scene_3 = [{'path': labeled_mix.next(), 'active': True, 'fold_filter': fold_id}]
                count_fold_total += Stitch(scene_3, args=_args, video_path=args.video_path,
                                           fold_count=args.fold_count).generate_jobs(
                    fold_id=fold_id,
                    count=2)

                # 2 people, 1 active
                scene_4 = [{'path': labeled_mix.current(), 'active': True, 'fold_filter': fold_id},
                           {'path': labeled_passive_mix.current(), 'active': False, 'fold_filter': fold_id}]
                count_fold_total += Stitch(scene_4, args=_args, video_path=args.video_path,
                                           fold_count=args.fold_count).generate_jobs(
                    fold_id=fold_id,
                    count=3)

                # 3 people, 1 active
                scene_5 = [{'path': labeled_mix.current(), 'active': True, 'fold_filter': fold_id},
                           {'path': labeled_passive_mix.current(), 'active': False, 'fold_filter': fold_id},
                           {'path': labeled_passive_mix.random(), 'active': False, 'fold_filter': fold_id}]
                count_fold_total += Stitch(scene_5, args=_args, video_path=args.video_path,
                                           fold_count=args.fold_count).generate_jobs(
                    fold_id=fold_id,
                    count=2)

                assert_verify_folds()

                if count_fold_total >= count_sequences_per_fold:
                    break

            if count_fold_total < count_sequences_per_fold:
                print("Only created %d sequences for this fold, even though %d were requested" % (
                    count_fold_total, count_sequences_per_fold))
