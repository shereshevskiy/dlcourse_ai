import numpy as np

from linear_classifer import softmax, cross_entropy_loss

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO_: Copy from previous assignment
    # raise Exception("Not implemented!")

    loss = (W * W).sum() * reg_strength
    grad = 2 * W * reg_strength

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO_ copy from the previous assignment
    # raise Exception("Not implemented!")

    preds = predictions.copy()

    probs = softmax(preds)

    loss = cross_entropy_loss(probs, target_index).mean()

    mask = np.zeros_like(preds)
    mask[np.arange(len(mask)), target_index] = 1
    # mask[target_index] = 1

    d_preds = - (mask - softmax(preds)) / mask.shape[0]

    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO_ copy from the previous assignment
        # raise Exception("Not implemented!")
        result = np.maximum(X, 0)
        self.X = X
        return result

    def backward(self, d_out):
        # TODO_ copy from the previous assignment
        # raise Exception("Not implemented!")
        d_X = (self.X > 0) * d_out
        return d_X

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO_ copy from the previous assignment
        # raise Exception("Not implemented!")
        W = self.W.value
        B = self.B.value
        self.X = X
        out = np.dot(X, W) + B
        return out

    def backward(self, d_out):
        # TODO_ copy from the previous assignment
        # raise Exception("Not implemented!")
        X = self.X
        W = self.W.value

        d_W = np.dot(X.T, d_out)
        d_B = np.dot(np.ones((X.shape[0], 1)).T, d_out)
        d_X = np.dot(d_out, W.T)

        self.W.grad += d_W
        self.B.grad += d_B

        return d_X

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.padding = padding

        self.stride = None
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = 0
        out_width = 0
        
        # TODO_: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops

        self.stride = 1
        s = self.stride
        out_height = int(1 + (height + 2 * self.padding - self.filter_size) / self.stride)
        out_width = int(1 + (width + 2 * self.padding - self.filter_size) / self.stride)

        pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        X = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)

        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        for oh in range(out_height):
            for ow in range(out_width):
                # TODO_: Implement forward pass for specific location
                for bs in range(batch_size):
                    for oc in range(self.out_channels):
                        out[bs, oh, ow, oc] = np.sum(X[bs, oh * s:oh * s + self.filter_size,
                                                           ow * s:ow * s + self.filter_size, :] *
                                                     self.W.value[:, :, :, oc]) + self.B.value[oc]

        self.X = X
        # raise Exception("Not implemented!")
        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        # initialization
        X = self.X
        W = self.W.value
        filter_size, filter_size, channels, out_channels = W.shape
        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        dX = np.zeros_like(X)
        dW = np.zeros_like(W)
        s = self.stride
        padding = self.padding

        # TODO_: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        dB = np.sum(d_out, (0, 1, 2))
        for n in range(batch_size):
            for c in range(channels):
                for oh in range(out_channels):
                    for ho in range(out_height):
                        for wo in range(out_width):
                            # TODO_: Implement backward pass for specific location
                            # Aggregate gradients for both the input and
                            # the parameters (W and B)
                            for hh in range(filter_size):
                                for ww in range(filter_size):
                                    dW[hh, ww, c, oh] += X[n, ho * s + hh, wo * s + ww, c] * d_out[n, ho, wo, oh]
                            for hi in range(height):
                                for wi in range(width):
                                    if (hi - ho * s >= 0) and (hi - ho * s < filter_size) and \
                                            (wi - wo * s >= 0) and (wi - wo * s < filter_size):
                                        dX[n, hi, wi, c] += W[hi - ho * s, wi - wo * s, c, oh] * d_out[n, ho, wo, oh]

        # raise Exception("Not implemented!")
        if padding != 0:
            dX = dX[:, padding:-padding, padding:-padding, :]  # bach to the initial input size

        self.B.grad = dB
        self.W.grad = dW
        return dX

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO_: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        # raise Exception("Not implemented!")

        out_height = int(np.ceil(1 + (height - self.pool_size) / self.stride))
        out_width = int(np.ceil(1 + (width - self.pool_size) / self.stride))
        out = np.zeros((batch_size, out_height, out_width, channels))
        s = self.stride
        for n in range(batch_size):
            for c in range(channels):
                for ho in range(out_height):
                    for wo in range(out_width):
                        out[n, ho, wo, c] = np.amax(X[n, ho * s:np.minimum(ho * s + self.pool_size, height),
                                                      wo * s:np.minimum(wo * s + self.pool_size, width), c])
        self.X = X
        return out


    def backward(self, d_out):
        # TODO_: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        # raise Exception("Not implemented!")

        X = self.X
        batch_size, height, width, channels = X.shape
        s = self.stride
        out_height = int(np.ceil(1 + (height - self.pool_size) / self.stride))
        out_width = int(np.ceil(1 + (width - self.pool_size) / self.stride))
        dX = np.zeros_like(X)

        for n in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        X_pool = X[n, oh * s:np.minimum(oh * s + self.pool_size, height),
                              ow * s:np.minimum(ow * s + self.pool_size, width), c]
                        dX_pool = np.zeros_like(X_pool)
                        ind_max = np.unravel_index(np.argmax(X_pool, axis=None), X_pool.shape)
                        dX_pool[ind_max] = 1
                        dX[n, oh * s:np.minimum(oh * s + self.pool_size, height),
                        ow * s:np.minimum(ow * s + self.pool_size, width), c] += dX_pool * d_out[n, oh, ow, c]

        return dX


    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO_: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        # raise Exception("Not implemented!")

        self.X = X
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # TODO_: Implement backward pass
        # raise Exception("Not implemented!")

        X = self.X
        dX = d_out.reshape(X.shape)

        return dX

    def params(self):
        # No params!
        return {}
