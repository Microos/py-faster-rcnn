from collections import OrderedDict
from traceback import print_exc
from ROC_FDDB.wider.test_wider import simple_test, set_gpu_id
from py_wider_eval import simple_wider_eval
from pylab import *
from find_gpu import find_avaiable_gpuid
from time import gmtime, strftime
import os

class OnTheFlyTester:
    def __set_result_dict_from_dir(self):
        #dirty function for debug only, never use this while deploying
        for r,ds,fs in os.walk(self.save_dir):
            for f in fs:
                with open(r+'/'+f,'r') as f:
                    lines = [l.strip() for l in f.readlines()]
                    iter = int(lines[0].split('_')[1])
                    aps = []
                    for i in range(1,4):
                        aps.append(float(lines[i].split(': ')[1]))
                    self.results[iter] = aps
                    self.done_dict[iter] = ''

                    self.results = self.sort_dict_by_key(self.results)
                    self.done_dict = self.sort_dict_by_key(self.done_dict)

    def __set_results_dict4test(self):
        # dirty function for debug only, never use this while deploying
        self.results[10000] = [0.6, 0.7, 0.8]
        self.results[20000] = [0.55, 0.78, 0.6]
        self.results[30000] = [0.65, 0.79, 0.62]
        self.results[40000] = [0.68, 0.789, 0.60]
        self.results[50000] = [0.70, 0.796, 0.69]

    def sort_dict_by_key(self, D):
        ret_dict = OrderedDict()
        for k in sorted(D.keys()):
            ret_dict[k] = D[k]
        return ret_dict

    def find_iter_cfmodels(self):
        tmp_dict = {}
        for r, dirs, files in os.walk(self.cfmodel_output_dir):
            for f in files:
                if not f.endswith('.caffemodel'): continue
                iter = int(f.split('.')[0].split('_')[-1])
                model_path = os.path.join(r, f)
                if iter not in self.done_dict.keys():
                    tmp_dict[iter] = model_path
        for k in sorted(tmp_dict.keys()):
            self.wait_dict[k] = tmp_dict[k]
            print 'Found Task: [{}] {}'.format(k, tmp_dict[k])

    '''init'''
    def __init__(self, netdef, cfmodel_output_dir, model_name, save_dir, baseline_aps=None):
        if not os.path.exists(netdef):
            print "Test net: '{}' does not exist".format(netdef)
            #try to set test netdef automatically
            this_dir = os.path.dirname(__file__)
            netdef = os.path.join(this_dir, '../../models/{}/VGG16/faster_rcnn_end2end','test.prototxt')
            print 'Auto Set Test Net to: {}'.format(netdef)
            assert os.path.exists(netdef), 'Auto Set Test Net Failed: {}'.format(netdef)

        assert os.path.exists(cfmodel_output_dir), cfmodel_output_dir
        self.test_net = netdef

        time_now = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
        self.save_dir = os.path.join(save_dir, '{}#{}'.format(model_name, time_now))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print 'OnTheFly test results will be saved to: {}'.format(self.save_dir)

        self.model_name = model_name
        self.cfmodel_output_dir = cfmodel_output_dir

        self.done_dict = OrderedDict()
        self.wait_dict = OrderedDict()
        self.results = OrderedDict()
        self.baseline_aps = [0.883, 0.764, 0.373] if baseline_aps is None else baseline_aps

    def test(self, old_gpu_id):
        try:
            self.__do_test()
            self.__write_results()
            self.__plot_fig()
        except:
            print_exc()
            sys.stderr.write('Failed to test, abort.')
            sys.stderr.flush()
        set_gpu_id(old_gpu_id)

    def __do_test(self):
        self.find_iter_cfmodels()
        if len(self.wait_dict.keys()) == 0:
            print 'No new models found in {}, abort.'.format(self.cfmodel_output_dir)
            return

        gpuid = find_avaiable_gpuid(require_memo=2200, gpuid_map={0: 1, 1: 0})
        print 'Found Best GPU ID: {}'.format(gpuid)

        if gpuid == -1:
            sys.stderr.write('Failed to find any idle gpu, abort.')
            sys.stderr.flush()
            return
        for iter, model in self.wait_dict.iteritems():
            model_name = '{}_{}'.format(self.model_name, iter)
            status = simple_test(netdef=self.test_net, cfmodel=model,
                                 model_name=model_name, gpu_id=gpuid,
                                 title='Testing {} model of {} iterations.'.format(self.model_name, iter))

            if status is True:
                m = self.wait_dict.pop(iter)
                self.done_dict[iter] = m
            else:
                sys.stderr.write('Failed to test task (iter:{}){}, skip.'.format(iter, model))
                sys.stderr.flush()

            print 'Evaluating AP values...'
            aps = simple_wider_eval(model_name=model_name, save_txt_dir=self.save_dir)
            print 'Easy: {:.4f}, Median: {:.4f}, Hard: {:4f}'.format(*aps)
            self.results[iter] = aps
        self.results = self.sort_dict_by_key(self.results)

    def __plot_fig(self):

        fig = figure(figsize=(10, 10))
        x = self.results.keys()

        legends = ['easy', 'median', 'head']
        result_linewidth = 3
        result_linestyle = 'solid'
        result_colors = ['green', 'orange', 'red']
        y_easy = [self.results[i][0] for i in x]
        y_mid = [self.results[i][1] for i in x]
        y_hard = [self.results[i][2] for i in x]

        for leg, y, c in zip(legends, [y_easy, y_mid, y_hard], result_colors):
            plot(x, y, color=c, ls=result_linestyle, lw=result_linewidth, marker='o',
                 label='{:<7}(max:{:.3f})'.format(leg, np.max(y)))
            for xi, yi in zip(x, y):
                text(xi - 3, yi + 0.005, '{:.3f}'.format(yi),
                     bbox={'facecolor': 'white', 'alpha': 0.3, 'pad': 2}, color=c)

        baseline_linewidth = 3
        baseline_linestyle = 'dotted'
        baseline_colors = ['green', 'orange', 'red']
        y_base_easy = [self.baseline_aps[0] for i in x]
        y_base_mid = [self.baseline_aps[1] for i in x]
        y_base_hard = [self.baseline_aps[2] for i in x]

        for leg, y, c in zip(self.baseline_aps, [y_base_easy, y_base_mid, y_base_hard], baseline_colors):
            plot(x, y, color=c, ls=baseline_linestyle, lw=baseline_linewidth, label=str(leg))
            text(np.min(x), np.max(y) + 0.005, 'Baseline:{:.3f}'.format(np.max(y)),
                 bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2}, color=c)

        ylim(0, 1)
        yticks(np.arange(0, 1.001, 0.05))
        xticks(x)
        grid()
        legend(loc=4)
        title('{} test result, {} iterations'.format(self.model_name, np.max(x)))

        fig_file = os.path.join(self.save_dir, 'test_result#{}.png'.format(np.max(self.results.keys())))
        fig.savefig(fig_file, dpi=200)
        print 'Figure saved to {}'.format(fig_file)
        #show()

    def __write_results(self):
        txtfile = os.path.join(self.save_dir, 'test_results_all#{}.txt'.format(np.max(self.results.keys())))
        s = ''
        for k, v in self.results.iteritems():
            aps = ','.join([str(x) for x in v])
            s += '{}:{}\n'.format(k, aps)
        with open(txtfile, 'w') as f:
            f.write(s)
        print 'Results wrote to: {}'.format(txtfile)


# aps = simple_wider_eval(model_name='test',save_txt_dir='/home/ylxie/Desktop')
# print aps

if __name__ == '__main__':

    netdef = '/home/ylxie/Space/work/py-faster-rcnn-fc2/models/naive_wider_test.prototxt'
    save_dir = '/home/ylxie/Desktop/otf_tester'
    cfmodel_output_dir = '/home/ylxie/Space/work/py-faster-rcnn2/output/adam/train/'
    model_name = 'naive'
    t = OnTheFlyTester(netdef=netdef, save_dir=save_dir, cfmodel_output_dir=cfmodel_output_dir, model_name=model_name)
    t.test()
    pass
