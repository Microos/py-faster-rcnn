import numpy as np
import caffe


'''
layer {
  name: 'nnnnnnnnnn'
  type: 'Python'
  bottom: 'xxxxxxxxxx'
  top: 'xxxxxxxxxxx'
  python_param {
    module: 'roi_data_layer.L2_norm_layer'
    layer: 'L2NormLayer'
  }
}
'''


class L2NormLayer(caffe.Layer):
    def l2_forward(self, x):
        self.x = x
        # step1
        self.xp2 = self.x ** 2  # (N,C,W,H)

        # step2
        self.sum_xp2 = np.sum(self.xp2, axis=(0, 1))  # (W,H)

        # step3
        self.sqrt_sum_xp2 = np.sqrt(self.sum_xp2+self.eps)  # (W,H)

        # step4
	#print "self.sum_xp2.min = {}".format(self.sum_xp2.min())
        #print "self.sqrt_sum_xp2.min = {}\n".format(self.sqrt_sum_xp2.min())
	xhat = self.x / self.sqrt_sum_xp2
	
        return xhat

    def l2_backward(self,dl):
        # step4
        d_x1 = dl / self.sqrt_sum_xp2
        d_sqrt_sum_xp2 = -np.sum(self.x * dl, axis=(0, 1)) / (self.eps+self.sqrt_sum_xp2 ** 2)

        # step3
        d_sum_xp2 = d_sqrt_sum_xp2 / (self.eps+2 * np.sqrt(self.sum_xp2))

        # step2
        d_xp2 = np.ones_like(self.xp2) * d_sum_xp2

        # step1
        d_x2 = 2 * self.x * d_xp2

        # step0
        d_x = d_x1 + d_x2
        return d_x

    def setup(self, bottom, top):
        assert len(top) == 1
        assert len(bottom) == 1
        self.x = 0
        self.xp2 = 0
        self.sum_xp2 = 0
        self.sqrt_sum_xp2 = 0
	self.eps = 1e-12

    def reshape(self, bottom, top):
        '''reshape during forward'''
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        out = self.l2_forward(bottom[0].data)
        top[0].data[...] = out

    def backward(self, top, propagate_down, bottom):
        out = self.l2_backward(top[0].diff)
        bottom[0].diff[...] = out
