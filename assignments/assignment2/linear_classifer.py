import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO_ implement softmax

    # raise Exception("Not implemented!")
    predictions -= np.max(predictions)
    res = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    # res = np.exp(predictions) / np.sum(np.exp(predictions))
    return res
    # Your final implementation shouldn't have any loops



def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # raise Exception("Not implemented!")
    res = - np.log(probs[np.arange(len(probs)), target_index])
    # res = - np.log(probs[target_index])

    return res


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
    # TODO_ implement softmax with cross-entropy
    # raise Exception("Not implemented!")
    predictions = predictions.copy()

    probs = softmax(predictions)

    loss = cross_entropy_loss(probs, target_index).mean()

    mask = np.zeros_like(predictions)
    mask[np.arange(len(mask)), target_index] = 1
    # mask[target_index] = 1

    dprediction = - (mask - softmax(predictions)) / mask.shape[0]


    return loss, dprediction


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

    # TODO: implement l2 regularization and gradient
    # raise Exception("Not implemented!")
    loss = (W*W).sum() * reg_strength
    grad = 2 * W * reg_strength


    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # raise Exception("Not implemented!")
    loss = softmax_with_cross_entropy(predictions, target_index)[0]
    dprediction = softmax_with_cross_entropy(predictions, target_index)[1]
    dW = np.dot(X.T, dprediction)


    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self, verbose = False):
        self.W = None
        self.verbose = verbose

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        W = self.W
        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # raise Exception("Not implemented!")

            batch_num = np.random.randint(len(batches_indices))
            # batch_num = 0
            X_batches = X[batches_indices[batch_num]]
            target_index = y[batches_indices[batch_num]]
            loss, dW = linear_softmax(X_batches, W, target_index)
            l2_loss, l2_dW = l2_regularization(W, reg_strength=reg)
            loss += l2_loss
            dW += l2_dW
            W -= learning_rate * dW
            loss_history.append(loss)

            # end
            if self.verbose:
                print("Epoch %i, loss: %f" % (epoch, loss))

        self.W = W
        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # raise Exception("Not implemented!")
        predictions = np.dot(X, self.W)
        probs = softmax(predictions)
        y_pred = np.argmax(probs, axis=1)


        return y_pred

