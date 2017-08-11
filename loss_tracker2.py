#!/usr/bin/env python
import matplotlib.pylab as plt
import numpy as np
import re
from scipy import interpolate


def get_log_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines


def get_iter_loss(data):
    iter = []
    loss = []
    for i in data:
        sp = i.split(':')
        iter.append(int(sp[0]))
        loss.append(float(sp[1]))
    return iter, loss

def median(midlist):
    midlist.sort()
    lens = len(midlist)
    if lens % 2 != 0:
        midl = (lens / 2)
        res = midlist[midl]
    else:
        odd = (lens / 2) - 1
        ev = (lens / 2)
        res = float(midlist[odd] + midlist[ev]) / float(2)
    return res


def cal_mid_line(iter, loss, method='median'):
    mid_line_iter = []
    mid_line_loss = []
    window_size = 10
    step_size = window_size * 3
    lo = 0

    iter_len = len(iter)

    while 1:
        hi = lo + window_size
        if hi > iter_len:
            if len(mid_line_loss) > 3 and len(mid_line_loss) != 0:
                tck = interpolate.splrep(mid_line_iter, mid_line_loss)
                mid_line_iter = iter[::iterplate_iterval]
                mid_line_loss = interpolate.splev(mid_line_iter, tck)

            return mid_line_iter, mid_line_loss
        # else
        mid_line_iter.append(iter[lo])
        points = loss[lo:hi]
        if method == 'median':
            l = median(points)
        elif method == 'mean':
            l = np.mean(points)
        else:
            raise NotImplementedError('undefined method: {}'.format(method))
        mid_line_loss.append(l)

        lo += step_size


def plot_loss(iter, loss, update_times=0):
    plt.ion()
    plt.clf()
    mid_iter, mid_loss = cal_mid_line(iter, loss, method)

    plt.plot(mid_iter, mid_loss, 'r-', linewidth=3)
    # plt.plot(iter, np.ones(len(iter)),'g.',linewidth=1)
    # plt.plot(iter, np.ones(len(iter))*0.5, 'g.', linewidth=1)

    plt.plot(iter, loss, 'b-', alpha=0.4)
    plt.title("Iteration:{}   UpdateTimes:{} ({}s)".format(max(iter), update_times, update_interval))

    plt.xlabel("iterations")
    plt.ylabel("loss")

    plt.yticks(np.linspace(0, max(loss) * 1.05, 20))
    plt.show()
    plt.savefig('/tmp/fi0g.png')
    for i in range(update_interval):
        plt.grid()
        plt.title("Iteration:{}   UpdateTimes:{} ({}s)".format(max(iter), update_times, update_interval - i))
        plt.pause(1)
        plt.grid()


if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) == 2:
        file_dir = sys.argv[1]
    else:
        file_dir = 'fwd_pls_relu_conv3_ext_pool'

    fn = 'over_all_loss.txt'


    fn=  os.path.join(os.path.dirname(__file__),'experiments','logs',file_dir, 'over_all_loss.txt')
    update_interval = 10
    method = 'mean'
    iterplate_iterval = 50
    i = 0
    while True:
        i += 1
        data = get_log_data(fn)
        iter, loss = get_iter_loss(data)
        plot_loss(iter, loss, i, )
