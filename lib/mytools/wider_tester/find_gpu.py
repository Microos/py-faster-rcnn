import re
import subprocess
import time
from traceback import print_exc

import numpy as np

'''
requirement:
    gpustat   // sudo pip install gpustat
'''

'''
Use 'gpustat' to find a gpu with enough memory for testing
'''


def __get_current_gpu_info():
    try:
        output = subprocess.check_output('gpustat')
        output = re.sub(u'\u001b\[.*?[@-~]', '', output)

    except:
        #print_exc()
        print 'Failed to invoke "gpustat"'
        return -1,-1,-1
    splits = output.split('\n')
    splits = [s for s in splits if s.startswith('[')]

    card_id = []
    rest_memo = []
    rest_usage = []

    for sp in splits:
        lines_sp = sp.split('|')
        card_id.append(int(sp[1]))

        usage = lines_sp[1].split(',')[1].replace('%', '').replace(' ', '')
        rest_usage.append(100 - int(usage))

        memos = [s.replace(' ', '').replace('MB', '') for s in lines_sp[2].split('/')]
        rest_memo.append(int(memos[1]) - int(memos[0]))
    return card_id, rest_memo, rest_usage


def find_avaiable_gpuid(require_memo=2500, gpuid_map=None, repeat_times=5, repeat_interval=0.2):
    # @require_memo expected memory for test in MB
    # @gpuid_map is used for multicard and there ids are not correctly displayed on nvidia-smi
    # if you set gpuid to 0 for traning, but get card 2 loaded, you can give a map: {0:2}
    # telling that using gpu0 are actually using gpu2

    has_enough_memo = None
    sum_usage = None

    for i in range(repeat_times):
        card_id, rest_memo, rest_usage = __get_current_gpu_info()
        if card_id == -1: return -1
        if sum_usage is None:
            sum_usage = [0 for _ in range(len(card_id))]
            has_enough_memo = [True for _ in range(len(card_id))]

        for j in range(len(card_id)):
            sum_usage[j] += rest_usage[j]
            has_enough_memo[j] = True if rest_memo[j] >= require_memo else False

        if i != repeat_times - 1: time.sleep(repeat_interval)

    selected_gpuid = []
    selected_usage = []

    for i in range(len(card_id)):
        if has_enough_memo[i]:
            selected_gpuid.append(card_id[i])
            selected_usage.append(sum_usage[i] / repeat_times)

    if len(selected_gpuid) == 0: return -1

    gpuid = selected_gpuid[np.argmax(selected_usage)]
    if gpuid_map is not None and gpuid in gpuid_map.keys():
        gpuid = gpuid_map[gpuid]

    return gpuid
    #return 2

if __name__ == '__main__':
    print find_avaiable_gpuid(gpuid_map={0: 1, 1: 0})
