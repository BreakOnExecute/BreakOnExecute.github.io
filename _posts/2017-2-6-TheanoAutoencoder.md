---
layout: post
title: Autoencoder in Theano
date: 2017-02-06 18:00:00
tags: Autoencoder Theano NeuralNetwork DeepLearning
description: Implementation of a 1 hidden layer autoencoder with tied weights in Theano
---



I implemented a simple autoencoder across three different platforms in Python.  This is the Theano version.


# Neural Network Autoencoder in Theano

This is an example of an autoencoder written in Theano. It has 1 hidden layer of a given size. I have tied the weights of the hidden->output layer to that of the input->hidden layer. I won’t go into what an autoencoder is. The link to the example data file I use for testing is [here](/data/donut_corr2.npy).

First, import various needed libraries, including Theano.


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import theano
import theano.tensor as T

```

# Define the Autoencoder class

Pass in the number of hidden layer units and an autoencoder ID.  The autoencoder ID is not used here, but could be used to help with debug signals if you stack autoencoders and there are problems in Theano.


```python
class AutoEncoder(object):
    # M is number of hidden units, ae_id is an id in case you want to stack
    # these later and have meaningful debug names in theano
    def __init__(self, M, ae_id):
        self.M = M
        self.id = ae_id

    # generate gaussian weights with shape = shape
    def init_weights(self, shape):
        return np.random.randn(*shape) / np.sqrt(np.sum(shape))

    # fit function that uses gradient descent with momentum
    # X = data
    # epochs = # of training epochs
    # lr = learning rate
    # momentum = momentum to use during learning
    def fit(self, X, epochs=10, lr=0.1, momentum=0.9):
        # setup all the shared variables
        # those prefixed with 'd' are used for momentum in gradient descent
        N, D = X.shape
        W_init = self.init_weights((D, self.M))
        self.W = theano.shared(W_init, name='W_%s' % self.id)
        self.bh = theano.shared(np.zeros(self.M), name='bh_%s' % self.id)
        self.bo = theano.shared(np.zeros(D), name='bo_%s' % self.id)
        self.dW = theano.shared(np.zeros(W_init.shape), name='dW_%s' % self.id)
        self.dbh = theano.shared(np.zeros(self.M), name='dbh_%s' % self.id)
        self.dbo = theano.shared(np.zeros(D), name='dbo_%s' % self.id)
        self.params = [self.W, self.bh, self.bo]
        self.dparams = [self.dW, self.dbh, self.dbo]
        # set the input variable
        # create functions to encode and decode the input data
        Xtrain = T.fmatrix(name='Xtrain_%s' % self.id)
        X_decoded = self.forward(Xtrain)
        X_encoded = self.forward_encode_only(Xtrain)
        self.decoded = theano.function(inputs=[Xtrain], outputs=X_decoded)
        self.encoded = theano.function(inputs=[Xtrain], outputs=X_encoded)
        # create cost function - mean squared error
        cost = ((Xtrain - X_decoded)**2).mean()
        cost_op = theano.function(inputs=[Xtrain], outputs=cost)
        # set the list of updates and create the Theano training function
        updates = [(p, p + momentum*dp - lr*T.grad(cost,p)) for p, dp in zip(self.params, self.dparams)] + \
                  [(dp, momentum*dp - lr*T.grad(cost,p)) for p, dp in zip(self.params, self.dparams)]
        train_op = theano.function(inputs=[Xtrain], updates=updates, outputs=cost)
        # create empty list to hold the costs at each epoch
        self.costs = []
        for i in xrange(epochs):
            # train and print out the cost
            self.costs.append(train_op(X))
            if i % 500 == 0:
                print 'Epoch %i of %i, cost = %s' % (i, epochs, self.costs[-1])

    # function to plot out the incoming data, the encoded data,
    # the decoded data, and the cost function vs. epochs
    def plot_debug(self, X, Y):
        encoder_output = self.encoded(X)
        predicted = self.decoded(X)
        plt.subplot(2, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
        plt.subplot(2, 2, 2)
        plt.scatter(encoder_output[:, 0], encoder_output[:, 1], c=Y, s=100, alpha=0.5)
        plt.subplot(2, 2, 3)
        plt.scatter(predicted[:, 0], predicted[:, 1], c=Y, s=100, alpha=0.5)
        plt.subplot(2, 2, 4)
        plt.plot(self.costs)
        plt.show()

    def forward_encode_only(self, X):
        encoded = T.nnet.sigmoid(X.dot(self.W) + self.bh)
        return encoded

    def forward(self, X):
        encoded = T.nnet.sigmoid(X.dot(self.W) + self.bh)
        # output is not sigmoid, just linear transform
        decoded = encoded.dot(self.W.T) + self.bo
        return decoded
```

# Test function

This function simply loads an simple example data file with 4 features (2 of them are correlated to another). Ideally we should be able to run this through an autoencoder and reproduce most of the data with only 2 columns.


```python
def TestAutoEncoderTheano():
    # Load simple example data, randomize the order, and then scale every
    # feature to between 0 & 1.  This example is 2 donuts at different radii
    # (and noise around them) with 2 columns that are correlated.  An
    # autoencoder (among other techniques) should be able to reduce this down
    # to 2 significant features instead of the original 4.
    data = np.load('../../data/donut_corr2.npy')
    data = shuffle(data, random_state=147)  # random_state set for testing
    Xtrain = data[:, :-1].astype(np.float32)
    Ytrain = data[:, -1].astype(np.int64)
    Xtrain = MinMaxScaler((0, 1)).fit_transform(Xtrain)
    # setup the AutoEncoder and fit the data, then plot debug charts
    ae = AutoEncoder(M=2, ae_id=1)
    ae.fit(Xtrain, lr=0.1, momentum=0.9, epochs=10000)
    ae.plot_debug(Xtrain, Ytrain)
```

# Test the autoencoder

Let's see this thing in action.  I didn't spend much time getting this even better, but I was able to reproduce the input pretty well with only 2 features, which is the expected outcome.


```python
TestAutoEncoderTheano()
```

    Epoch 0 of 10000, cost = 0.426908395353
    Epoch 500 of 10000, cost = 0.000860154586943
    Epoch 1000 of 10000, cost = 0.000350999087128
    Epoch 1500 of 10000, cost = 0.000231595223452
    Epoch 2000 of 10000, cost = 0.000183510693309
    Epoch 2500 of 10000, cost = 0.000159628667921
    Epoch 3000 of 10000, cost = 0.000146192322934
    Epoch 3500 of 10000, cost = 0.0001380129861
    Epoch 4000 of 10000, cost = 0.00013276838868
    Epoch 4500 of 10000, cost = 0.000129280369577
    Epoch 5000 of 10000, cost = 0.000126894862359
    Epoch 5500 of 10000, cost = 0.000125224837335
    Epoch 6000 of 10000, cost = 0.000124030584438
    Epoch 6500 of 10000, cost = 0.000123158569967
    Epoch 7000 of 10000, cost = 0.000122507917814
    Epoch 7500 of 10000, cost = 0.000122011017775
    Epoch 8000 of 10000, cost = 0.000121621831439
    Epoch 8500 of 10000, cost = 0.000121308605391
    Epoch 9000 of 10000, cost = 0.000121049209033
    Epoch 9500 of 10000, cost = 0.000120828086046



![png](/jupyter_stuff/TheanoAutoencoder/TheanoAutoencoder_files/TheanoAutoencoder_8_1.png)

