__author__ = 'YiLiangXie'
import os
from time import gmtime, strftime
from traceback import print_exc

import numpy as np


class LossWritter():
    def __init__(self, solver, model_name, loss_blob_name_list=None, loss_blob_group_dict=None, given_dir=None,
                 over_all_loss=True, disp_interval=10):
        self.solver = solver
        self.net = solver.net
        self.loss_dir = self.make_loss_dir(model_name, given_dir)
        self.loss_blob_name_list = loss_blob_name_list
        self.loss_blob_group_dict = loss_blob_group_dict
        self.check_blob_names()
        self.disp_interval = disp_interval #display save info every $disp_interval times call self.log_loss()
        self.log_times = 0

    def make_loss_dir(self, model_name, given_dir):


        if given_dir is None:
            use_default_dir = True
        else:
            if os.path.exists(given_dir):
                use_default_dir = False
            else:
                use_default_dir = True

        if use_default_dir:
            this_dir = os.path.dirname(__file__)
            time_now = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
            model_dir_name_with_time = '{}#{}'.format(model_name, time_now)
            model_dir_path_with_time = os.path.realpath(os.path.join(this_dir, '../..', 'experiments', 'logs',model_dir_name_with_time))
            model_dir_path_wo_time = os.path.realpath(os.path.join(this_dir, '../..', 'experiments', 'logs',model_name))
            if not os.path.exists(model_dir_path_wo_time):
                with_time = False
            else: #already existed
                if len(os.listdir(model_dir_path_wo_time)) == 0:
                    with_time = False
                else:
                    with_time = True

            loss_dir = model_dir_path_with_time if with_time else model_dir_path_wo_time

        else:
            loss_dir = os.path.join(given_dir, model_name)

        if os.path.exists(loss_dir):
            return loss_dir

        try:
            os.makedirs(loss_dir)
            print 'LossWritter:: Loss will be saved to: {}'.format(loss_dir)
            return loss_dir
        except:
            print_exc()
            raise Exception('LossWritter:: Failed to make dir at: {}'.format(loss_dir))

    def check_blob_names(self):
        def auto_find_blob_names():
            # auto find blobs with 'loss' in names
            return [n for n in self.net.blobs if 'loss' in n]

        if self.loss_blob_name_list is None:
            self.loss_blob_name_list = auto_find_blob_names()
        else:
            for k in self.loss_blob_name_list:
                assert k in self.net.blobs.keys()

        if self.loss_blob_group_dict is not None:
            max_len = len(self.net.blobs.keys())
            for k, v in self.loss_blob_group_dict.iteritems():
                for vi in v:
                    # check if index exceeds the bound
                    if vi < 0 or vi >= max_len:
                        print "LossWritter:: index {} in argument loss_blob_group_dict['{}']={} out of bound {}, " \
                              "\n\tthis argument was canceled and will have no effect.".format(vi, k, v, max_len)
                        self.loss_blob_group_dict = None
                        break
        if self.loss_blob_group_dict is not None:
            self.loss_blob_group_dict['over_all_loss'] = range(len(self.loss_blob_name_list))
        else:
            self.loss_blob_group_dict = {'over_all_loss' : range(len(self.loss_blob_name_list))}

    def log_loss(self):
        self.log_times += 1
        # loss_list ['rpn_cls_loss','rpn_loss_bbox','loss_cls','loss_bbox']
        # loss_group_dict {'rpn_loss':[0,1], 'fc_loss',[2,3]}
        self.__write_loss(self.loss_blob_name_list)

        if self.loss_blob_group_dict is not None:
            iter = self.solver.iter
            for k, v in self.loss_blob_group_dict.iteritems():
                filename = os.path.join(self.loss_dir, '{}.txt'.format(k))
                loss = np.sum([self.net.blobs[self.loss_blob_name_list[i]].data for i in v])
                with open(filename, 'a') as f:
                    f.write("{}:{}\n".format(iter, loss))
                if self.log_times % self.disp_interval == 0: print 'LossWritter:: {} appended to {}'.format(k, filename)

    def __write_loss(self, names):
        iter = self.solver.iter
        for name in names:
            loss = self.net.blobs[name].data
            filename = os.path.join(self.loss_dir, '{}.txt'.format(name))
            with open(filename, 'a') as f:
                f.write("{}:{}\n".format(iter, loss))
            if self.log_times % self.disp_interval == 0: print 'LossWritter:: {} appended to {}'.format(name, filename)


if __name__ == '__main__':
    pass
