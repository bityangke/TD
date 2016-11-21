# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from tdcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_videos = len(roidb)
    # Sample random scales to use for each video in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_videos)
    assert(cfg.TRAIN.BATCH_SIZE % num_videos == 0), \
        'num_videos ({}) must divide BATCH_SIZE ({})'. \
        format(num_videos, cfg.TRAIN.BATCH_SIZE)
    rois_per_video = cfg.TRAIN.BATCH_SIZE / num_videos
    fg_rois_per_video = np.round(cfg.TRAIN.FG_FRACTION * rois_per_video)

    # Get the input video blob, formatted for caffe
    video_blob, video_scales = _get_video_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt windows: (x1, x2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_windows = np.empty((len(gt_inds), 3), dtype=np.float32)
        gt_windows[:, 0:2] = roidb[0]['windows'][gt_inds, :]
        gt_windows[:, -1] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_windows'] = gt_windows
    else: # not using RPN
        assert not cfg.TRAIN.HAS_RPN, "Not Implement Yet"
#        # Now, build the region of interest and label blobs
#        rois_blob = np.zeros((0, 5), dtype=np.float32)
#        labels_blob = np.zeros((0), dtype=np.float32)
#        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
#        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
#        # all_overlaps = []
#        for im_i in xrange(num_images):
#            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
#                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
#                               num_classes)
#
#            # Add to RoIs blob
#            rois = _project_im_rois(im_rois, im_scales[im_i])
#            batch_ind = im_i * np.ones((rois.shape[0], 1))
#            rois_blob_this_image = np.hstack((batch_ind, rois))
#            rois_blob = np.vstack((rois_blob, rois_blob_this_image))
#
#            # Add to labels, bbox targets, and bbox loss blobs
#            labels_blob = np.hstack((labels_blob, labels))
#            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
#            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
#            # all_overlaps = np.hstack((all_overlaps, overlaps))
#
#        # For debug visualizations
#        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)
#
#        blobs['rois'] = rois_blob
#        blobs['labels'] = labels_blob
#
#        if cfg.TRAIN.BBOX_REG:
#            blobs['bbox_targets'] = bbox_targets_blob
#            blobs['bbox_inside_weights'] = bbox_inside_blob
#            blobs['bbox_outside_weights'] = \
#                np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs

#def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
#    """Generate a random sample of RoIs comprising foreground and background
#    examples.
#    """
#    # label = class RoI has max overlap with
#    labels = roidb['max_classes']
#    overlaps = roidb['max_overlaps']
#    rois = roidb['windows']
#
#    # Select foreground RoIs as those with >= FG_THRESH overlap
#    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
#    # Guard against the case when an image has fewer than fg_rois_per_image
#    # foreground RoIs
#    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
#    # Sample foreground regions without replacement
#    if fg_inds.size > 0:
#        fg_inds = npr.choice(
#                fg_inds, size=fg_rois_per_this_image, replace=False)
#
#    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
#    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
#                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
#    # Compute number of background RoIs to take from this image (guarding
#    # against there being fewer than desired)
#    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
#    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
#                                        bg_inds.size)
#    # Sample foreground regions without replacement
#    if bg_inds.size > 0:
#        bg_inds = npr.choice(
#                bg_inds, size=bg_rois_per_this_image, replace=False)
#
#    # The indices that we're selecting (both fg and bg)
#    keep_inds = np.append(fg_inds, bg_inds)
#    # Select sampled values from various arrays:
#    labels = labels[keep_inds]
#    # Clamp labels for the background RoIs to 0
#    labels[fg_rois_per_this_image:] = 0
#    overlaps = overlaps[keep_inds]
#    rois = rois[keep_inds]
#
#    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
#            roidb['bbox_targets'][keep_inds, :], num_classes)
#
#    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _get_video_blob(roidb, scale_inds):
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """
    num_videos = len(roidb)
    processed_videos = []
    video_scales = []
    for i in xrange(num_videos):
      start = roidb[i]['start_frame']
      video = np.zeros((cfg.TRAIN.LENGTH, cfg.TRAIN.CROP_SIZE[0],
                        cfg.TRAIN.CROP_SIZE[1], 3))
      for j in xrange(cfg.TRAIN.LENGTH):
        frame = cv2.imread(roidb[i]['video'] % str(start + j).zfill(5))
        frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, cfg.TRAIN.FRAME_SIZE,
                                        cfg.TRAIN.CROP_SIZE)
        if roidb[i]['flipped']:
            frame = frame[:, ::-1, :]
        video[j] = frame
      processed_videos.append(video)

    # Create a blob to hold the input images
    blob = video_list_to_blob(processed_videos)

    return blob

#def _project_im_rois(im_rois, im_scale_factor):
#    """Project image RoIs into the rescaled training image."""
#    rois = im_rois * im_scale_factor
#    return rois
#
#def _get_bbox_regression_labels(bbox_target_data, num_classes):
#    """Bounding-box regression targets are stored in a compact form in the
#    roidb.
#
#    This function expands those targets into the 4-of-4*K representation used
#    by the network (i.e. only one class has non-zero targets). The loss weights
#    are similarly expanded.
#
#    Returns:
#        bbox_target_data (ndarray): N x 4K blob of regression targets
#        bbox_inside_weights (ndarray): N x 4K blob of loss weights
#    """
#    clss = bbox_target_data[:, 0]
#    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
#    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
#    inds = np.where(clss > 0)[0]
#    for ind in inds:
#        cls = clss[ind]
#        start = 4 * cls
#        end = start + 4
#        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
#        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
#    return bbox_targets, bbox_inside_weights
#
#def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
#    """Visualize a mini-batch for debugging."""
#    import matplotlib.pyplot as plt
#    for i in xrange(rois_blob.shape[0]):
#        rois = rois_blob[i, :]
#        im_ind = rois[0]
#        roi = rois[1:]
#        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
#        im += cfg.PIXEL_MEANS
#        im = im[:, :, (2, 1, 0)]
#        im = im.astype(np.uint8)
#        cls = labels_blob[i]
#        plt.imshow(im)
#        print 'class: ', cls, ' overlap: ', overlaps[i]
#        plt.gca().add_patch(
#            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
#                          roi[3] - roi[1], fill=False,
#                          edgecolor='r', linewidth=3)
#            )
#        plt.show()
