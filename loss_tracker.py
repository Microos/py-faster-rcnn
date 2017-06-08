#!/usr/bin/env python
import matplotlib.pylab as plt
import numpy as np
import re





def get_log_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [l for l in lines if 'Im' != l[0:2] ]
    return ''.join(lines)

def get_iter_loss(data):
    loss = re.findall(' loss = [0-9]*.[0-9]*\n', data)
    iter = re.findall('] Iteration [0-9]*', data)
    loss = [float(i.strip(' loss = ')) for i in loss]
    iter = [int(i.strip('] Iteration ')) for i in iter][::2]
    return iter,loss

def median(midlist):
    midlist.sort()
    lens = len(midlist)
    if lens % 2 != 0:
        midl = (lens / 2)
        res = midlist[midl]
    else:
        odd = (lens / 2) -1
        ev = (lens / 2)
        res = float(midlist[odd] + midlist[ev]) / float(2)
    return res

def cal_mid_line(iter,loss):
    mid_iter = []
    mid_loss = []

    iter_window_size = 1000/20
    step_size= 800/20
    lower_idx = 0

    lens = len(iter)
    while True:
        if(lower_idx >= lens):
            upper_idx = lens
        upper_idx = lower_idx + iter_window_size
        sub_loss = loss[lower_idx: upper_idx]
        m = median(sub_loss)
        mid_iter.append(iter[lower_idx]+iter_window_size*10)
        mid_loss.append(m)

        lower_idx = lower_idx+step_size
        if(upper_idx >= lens):
            break
    # sub_loss = loss[-window_size:]
    # m = median(sub_loss)
    # mid_iter.append(iter[-1])
    # mid_loss.append(m)
    return mid_iter, mid_loss


def plot_loss(iter,loss,update_times=0, update_interval=20):
    plt.ion()
    plt.clf()
    mid_iter, mid_loss = cal_mid_line(iter, loss)

    plt.plot(mid_iter, mid_loss, 'r-', linewidth=3)
    plt.plot(iter, np.ones(len(iter)),'g.',linewidth=1)
    # plt.plot(iter, np.ones(len(iter))*0.5, 'g.', linewidth=1)

    plt.plot(iter, loss, 'b-', alpha=0.4)
    plt.title("Iteration:{}   UpdateTimes:{} ({}s)".format(max(iter), update_times, update_interval))

    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.yticks(np.arange(0, 10, 0.5))
    plt.show()
    plt.savefig('/tmp/fig.png')
    for i in range(update_interval):
        plt.grid()
        plt.title("Iteration:{}   UpdateTimes:{} ({}s)".format(max(iter), update_times, update_interval-i))
        plt.pause(1)
        plt.grid()


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        fn = sys.argv[1]

    else:
        fn = 'pycharm_log.log'

    # fn = '/home/rick/Space/clone/py-faster-rcnn/experiments/logs/faster_rcnn_end2end_VGG16_.txt.2017-04-20_10-31-36'
    i = 0
    while True:
        i+=1
        data = get_log_data(fn)
        iter, loss = get_iter_loss(data)
        plot_loss(iter,loss,i)

    raw_input('Enter to stop')
