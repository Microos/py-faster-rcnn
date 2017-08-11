# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        try:
            layer_params = yaml.load(self.param_str_)
        except:
            layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        # labels
        top[1].reshape(1, 1)
        # bbox_targets
        top[2].reshape(1, self._num_classes * 4)
        # bbox_inside_weights
        top[3].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        top[4].reshape(1, self._num_classes * 4)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets

        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

        if DEBUG:
            gt_bbox = gt_boxes[:, :4]
            pps_bbox = all_rois[:, 1:5]
            gt_bbox_info_str = _get_gt_bbox_info(gt_bbox)
            print gt_bbox_info_str
            rc_info_str = _get_proposal_recall_info(gt_bbox, pps_bbox, overlaps)
            print rc_info_str

        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois, gt_boxes, overlaps, fg_rois_per_image,
            rois_per_image, self._num_classes)

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _get_gt_bbox_info(gt_bbox):
    shp_gt = gt_bbox.shape
    n_gt = shp_gt[0]
    # bbox.area: min,mean,max
    wh = np.zeros(shape=(n_gt, 2))
    wh[:, 0] = gt_bbox[:, 2] - gt_bbox[:, 0]
    wh[:, 1] = gt_bbox[:, 3] - gt_bbox[:, 1]

    area = wh[:, 0] * wh[:, 1]
    _min_ = area.min()
    _max_ = area.max()
    _mean_ = area.mean()

    info = 'num_gt:{}; gt_area: min: {:.2f}, max: {:.2f}, mean: {:.2f}'.format(n_gt, _min_, _max_, _mean_)
    return info

def _get_proposal_recall_info(gt_bbox, pps_bbox, overlaps):
    # Calculate recall in ROC fashion
    # @ gt_bbox: n*4 (xmin,ymin,xmax,ymax)
    # @ pps_bbox: n*4 (xmin,ymin,xmax,ymax)
    # @ overlaps: e.g. [5900,20]

    shp_gt = gt_bbox.shape
    shp_pps = pps_bbox.shape
    n_gt = shp_gt[0]


    def recall_with_threas(thres):
        #recall_rate
        max_overlaps = overlaps.max(axis=0) #I am a box, I have the max overlaps
        recall_num = len(np.where(max_overlaps>thres)[0])
        recall_rate = recall_num/float(n_gt)

        #recall_specified
        idx = np.where(overlaps>thres)
        idx = np.vstack(idx).T
        box_specified_recall_num = np.zeros(shape=(shp_gt))
        used_pps_list = []
        for i in range(shp_pps):
            ppidx, gtidx = idx[i,:]
            if ppidx not in used_pps_list:
                used_pps_list.append(ppidx)
                box_specified_recall_num[gtidx]+=1

        return recall_rate, box_specified_recall_num
    info_str = '\n'
    info_str_template = '[t:{}] recall_rate: {:.3f}' \
                        '\n\teach_gt.fg_pps_num: min:{:.2f}, max:{:.2f}, mean:{:.2f}\n'
    thresholds = [0.1,0.3,0.5,0.7,0.9]
    for th in thresholds:
        rc_rate, b_sp_rc_num = recall_with_threas(th)
        idx = np.where(b_sp_rc_num > 0)[0]
        if len(idx) == 0:
            info_str += info_str_template.format(th,rc_rate,0,0,0)
        else:
            b_sp_rc_num = b_sp_rc_num[idx]
            _min_ = np.min(b_sp_rc_num)
            _max_ = np.max(b_sp_rc_num)
            _mean_ = np.mean(b_sp_rc_num)
            info_str += info_str_template.format(th,rc_rate,_min_,_max_,_mean_)

    return info_str

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        ins = ind.astype(np.int)
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind.astype(int), 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, overlaps, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)

    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
