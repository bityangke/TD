# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2

def video_list_to_blob(videos):
    """Convert a list of videos into a network input.

    Assumes videos are already prepared (means subtracted, BGR order, ...).
    """
    shape = videos[0].shape
    num_videos = len(videos)
    blob = np.zeros((num_videos, shape[0], shape[1], shape[2], 3),
                    dtype=np.float32)
    for i in xrange(num_videos):
        video = videos[i]
        blob[i] = video
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 4, 1, 2, 3)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, target_size, crop_size):
    """Mean subtract, resize and crop an frame for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, target_size, interpolation=cv2.INTER_LINEAR)
    im -= pixel_means
    x = np.random.randint(target_size[0] - crop_size[0] + 1)
    y = np.random.randint(target_size[1] - crop_size[1] + 1)
    return im[x:x+crop_size[0], y:y+crop_size[1], :]
