import numpy as np

from assignments.assignment2.layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization
from assignments.assignment1.linear_classifer import softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO_ Create necessary layers
        # raise Exception("Not implemented!")

        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_layer = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO_ Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        # raise Exception("Not implemented!")

        # initialization
        params = self.params()
        W1 = params["W1"]
        B1 = params["B1"]
        W2 = params["W2"]
        B2 = params["B2"]

        # the cleaning of gradients
        W1.grad = np.zeros_like(W1.value)
        B1.grad = np.zeros_like(B1.value)
        W2.grad = np.zeros_like(W2.value)
        B2.grad = np.zeros_like(B2.value)

        # forward pass
        out1 = self.layer1.forward(X)
        out_relu = self.relu_layer.forward(out1)
        out2 = self.layer2.forward(out_relu)
        loss, d_preds = softmax_with_cross_entropy(out2, y)

        # backward pass
        d_out2 = self.layer2.backward(d_preds)
        d_out_relu = self.relu_layer.backward(d_out2)
        d_out1 = self.layer1.backward(d_out_relu)

        # TODO_ Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # raise Exception("Not implemented!")

        # add regularization
        l2_W1_loss, l2_W1_grad = l2_regularization(W1.value, self.reg)
        l2_B1_loss, l2_B1_grad = l2_regularization(B1.value, self.reg)
        l2_W2_loss, l2_W2_grad = l2_regularization(W2.value, self.reg)
        l2_B2_loss, l2_B2_grad = l2_regularization(B2.value, self.reg)

        l2_reg = l2_W1_loss + l2_B1_loss + l2_W2_loss + l2_B2_loss
        loss += l2_reg

        W1.grad += l2_W1_grad
        B1.grad += l2_B1_grad
        W2.grad += l2_W2_grad
        B2.grad += l2_B2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO_: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        y_pred = np.zeros(X.shape[0], np.int)

        # raise Exception("Not implemented!")

        out1 = self.layer1.forward(X)
        out_relu = self.relu_layer.forward(out1)
        predictions = self.layer2.forward(out_relu)
        probs = softmax(predictions)
        y_pred = np.argmax(probs, axis=1)

        return y_pred

    def params(self):
        result = {
            "W1": self.layer1.params()["W"],
            "B1": self.layer1.params()["B"],
            "W2": self.layer2.params()["W"],
            "B2": self.layer2.params()["B"]
        }

        # TODO_ Implement aggregating all of the params

        # raise Exception("Not implemented!")

        return result
