# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os,sys
from numpy.random import standard_normal
from math import sqrt

from caffe.proto import caffe_pb2
import google.protobuf as pb2

def svd_orthonormal(shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """
    def lsuv_init(self):
        init_mode = cfg.TRAIN.LSUVINIT
        if init_mode not in ['Orthonormal','LSUV','OrthonormalLSUV']: return
        print '---------------{} INIT---------------'.format(init_mode)
        lyr_names = list(self.solver.net._layer_names)

        def filter_layer_type(layer_name):
            '''True if this layer is not activation or BN layer'''

            idx = lyr_names.index(layer_name)
            assert idx >= 0, idx
            layer = self.solver.net.layers
            return False if (layer[idx].type == 'ReLU' or layer[idx].type == 'BatchNorm') else True

        solver = self.solver
        margin = 0.02
        max_iter = 30 #30
        needed_variance = 1.0
        var_before_relu_if_inplace = True
        for k,v in solver.net.params.iteritems():
            #skip bn and activation
            if not filter_layer_type(k): continue

            #skip non-learnable layers
            try:
                print(k, v[0].data.shape)
            except:
                print 'Skipping layer ', k, ' as it has no parameters to initialize'
                continue

            if 'Orthonormal' in init_mode:
                weights = svd_orthonormal(v[0].data[:].shape)
                solver.net.params[k][0].data[:] = weights
            else:
                weights = solver.net.params[k][0].data[:]

            if 'LSUV' in init_mode:
                if var_before_relu_if_inplace:
                    solver.net.forward(end=k)
                else:
                    solver.net.forward()
                out_blob_name = solver.net.top_names[k]
                assert len(out_blob_name)==1, out_blob_name
                out_blob_name = out_blob_name[0]
                v = solver.net.blobs[out_blob_name]
                var1 = np.var(v.data[:])
                mean1 = np.mean(v.data[:])
                print k, 'var = ', var1, 'mean = ', mean1
                sys.stdout.flush()
                iter_num = 0

                while (abs(needed_variance - var1) > margin):
                    weights = solver.net.params[k][0].data[:]
                    solver.net.params[k][0].data[:] = weights / sqrt(var1)
                    if var_before_relu_if_inplace:
                        solver.net.forward(end=k)
                    else:
                        solver.net.forward()

                    v = solver.net.blobs[out_blob_name]
                    var1 = np.var(v.data[:])
                    mean1 = np.mean(v.data[:])
                    print(k, 'var = ', var1, 'mean = ', mean1)
                    sys.stdout.flush()
                    iter_num +=1

                    if iter_num > max_iter:
                        print 'Could not converge in ', iter_num, ' iterations, go to next layer'
                        break
            print ''

        print "Initialization finished!"
        solver.net.forward()
        for k, v in solver.net.blobs.iteritems():
            try:
                print(k, v.data[:].shape, ' var = ', np.var(v.data[:]), ' mean = ', np.mean(v.data[:]))
            except:
                print 'Skiping layer', k


        init_save_to = '{}_{}.caffemodel'.format(self.pretrained_model.split('.caffemodel')[0], init_mode)
        print "Saving {} init model....".format(init_mode)
        solver.net.save(init_save_to)
        print "Finished. Model saved to:", init_save_to
        print '---------------{} INIT DONE---------------'.format(init_mode)

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

        self.lsuv_init()
        print ''
    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('bbox_pred'))

        if scale_bbox_params:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (1 * self.solver_param.display) == 0:
                rest_time = timer.average_time * (max_iters - self.solver.iter)
                h = rest_time / 3600
                m = (rest_time % 3600) / 60
                print '\nspeed: {:.3f}s / iter; {}h {}m to go; {} left\n'.format(timer.average_time, int(h), int(m),
                                                                                 max_iters - self.solver.iter)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb

def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    roidb = filter_roidb(roidb)
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
