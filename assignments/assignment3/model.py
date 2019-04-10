import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )
from linear_classifer import softmax


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO_ Create necessary layers
        # raise Exception("Not implemented!")

        weight, height, cannels = input_shape
        filter_size = 3
        pool_size = 4
        padding = 1
        stride = pool_size

        self.conv1 = ConvolutionalLayer(cannels, conv1_channels, filter_size, padding)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(pool_size, stride)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size, padding)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(pool_size, stride)
        self.flatten = Flattener()
        n_fc_input = int(height / pool_size / pool_size * weight / pool_size / pool_size * conv2_channels)
        self.fc = FullyConnectedLayer(n_fc_input, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO_ Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        # raise Exception("Not implemented!")

        # initialization
        params = self.params()
        W1 = params["W1"]
        B1 = params["B1"]
        W2 = params["W2"]
        B2 = params["B2"]
        W3 = params["W3"]
        B3 = params["B3"]

        # the cleaning of params
        # W1.value = np.zeros_like(W1.value)
        # B1.value = np.zeros_like(B1.value)
        # W2.value = np.zeros_like(W2.value)
        # B2.value = np.zeros_like(B2.value)
        # W3.value = np.zeros_like(W3.value)
        # B3.value = np.zeros_like(B3.value)

        # the cleaning of gradients
        W1.grad = np.zeros_like(W1.value)
        B1.grad = np.zeros_like(B1.value)
        W2.grad = np.zeros_like(W2.value)
        B2.grad = np.zeros_like(B2.value)
        W3.grad = np.zeros_like(W3.value)
        B3.grad = np.zeros_like(B3.value)

        # forward pass
        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.maxpool1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.maxpool2.forward(out)
        out = self.flatten.forward(out)
        out = self.fc.forward(out)
        loss, d_preds = softmax_with_cross_entropy(out, y)

        # backward pass
        d_out = self.fc.backward(d_preds)
        d_out = self.flatten.backward(d_out)
        d_out = self.maxpool2.backward(d_out)
        d_out = self.relu2.backward(d_out)
        d_out = self.conv2.backward(d_out)
        d_out = self.maxpool1.backward(d_out)
        d_out = self.relu1.backward(d_out)
        d_out = self.conv1.backward(d_out)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        # raise Exception("Not implemented!")
        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.maxpool1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.maxpool2.forward(out)
        out = self.flatten.forward(out)
        out = self.fc.forward(out)
        probs = softmax(out)
        y_pred = np.argmax(probs, axis=1)

        return y_pred

    def params(self):
        result = {}

        # TODO_: Aggregate all the params from all the layers
        # which have parameters
        # raise Exception("Not implemented!")
        result = {
            "W1": self.conv1.params()["W"],
            "B1": self.conv1.params()["B"],
            "W2": self.conv2.params()["W"],
            "B2": self.conv2.params()["B"],
            "W3": self.fc.params()["W"],
            "B3": self.fc.params()["B"]
        }

        return result
