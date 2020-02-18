import argparse
import numpy as np
import pickle
import random
import gzip
import matplotlib.pyplot as plt
from IPython import embed

class NN:
    def __init__(self, sizes, keep_prob=-1):
        self.L = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(n, 1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n, m) for (m, n) in zip(self.sizes[:-1], self.sizes[1:])]
        self.keep_prob = keep_prob
        self.acc_train_array = []
        self.acc_test_array = []
        self.threshold = 20

    def sigmoid(self, z):
        """
        activation function
        """
        z = np.clip(z, -self.threshold, self.threshold)
        return 1.0 / (1.0 + np.exp(-z))


    def sigmoid_prime(self, z):
        """
        derivative of activation function
        """
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

    def forward_prop(self, a):
        """
        memory aware forward propagation for testing
        only.  back_prop implements it's own forward_prop
        """
        a_list = [a]
        z_list = [np.zeros(a.shape)]  # Pad with a placeholder so that indices match

        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, a) + b
            z_list.append(z)
            a = self.sigmoid(z)
            a_list.append(a)
        return a_list[-1]

    def grad_cost(self, a, y):
        """
        gradient of cost function
        Assumes C(a,y) = (a-y)^2/2
        """
        return (a - y)

    def SGD_train(self, train_x,train_y, epochs, eta, lam=0.0, verbose=True, test_x=None,test_y=None):
        """
        SGD for training parameters
        epochs is the number of epocs to run
        eta is the learning rate
        lam is the regularization parameter
        If verbose is set will print progressive accuracy updates
        If test set is provided, routine will print accuracy on test set as learning evolves
        """
        n_train = len(train_x)
        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            for kk in range(n_train):
                xk = train_x[perm[kk]]
                yk = train_y[perm[kk]]

                if self.keep_prob != -1:
                    dw,db = self.back_prop_dropout(xk,yk)
                else:
                    dw,db = self.back_prop(xk,yk)

                for ll in range(self.L - 1):
                    self.weights[ll] = self.weights[ll]*(1-lam*eta) - eta * dw[ll]
                    self.biases[ll] = self.biases[ll] - eta * db[ll]

            if verbose:
                if epoch == 0 or (epoch + 1) % 20 == 0:
                    acc_train = self.evaluate(train_x,train_y)
                    self.acc_train_array.append(acc_train)
                    if test_x is not None:
                        acc_test = self.evaluate(test_x,test_y)
                        self.acc_test_array.append(acc_test)
                        print("Epoch {:4d}: Train {:10.5f}, Test {:10.5f}".format(
                            epoch+1, acc_train, acc_test))
                    else:
                        print("Epoch {:4d}: Train {:10.5f}".format(
                            epoch+1, acc_train))

    def back_prop(self, x, y):
        """
        Back propagation for derivatives of C wrt parameters
        """
        db_list = [np.zeros(b.shape) for b in self.biases]
        dW_list = [np.zeros(W.shape) for W in self.weights]

        a = x
        a_list = [a]
        z_list = [np.zeros(a.shape)]  # Pad with a placeholder so that indices match

        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, a) + b
            z_list.append(z)
            a = self.sigmoid(z)
            a_list.append(a)

        # Back propagate deltas to compute derivatives
        # The following list gives hints on how to do it
        # calculating delta (Error) for the output layer
        # for the appropriate layers compute db_list[ell], dW_list[ell], delta

        L = self.L
        delta = [np.zeros((n, 1)) for n in self.sizes]
        delta[L - 1] = self.grad_cost(a_list[L - 1], y) * self.sigmoid_prime(z_list[L - 1])

        for ll in range(L - 1, 0, -1):
            db_list[ll - 1] = delta[ll]
            dW_list[ll - 1] = np.dot(delta[ll], a_list[ll - 1].T)
            delta[ll - 1] = np.dot(self.weights[ll - 1].T, delta[ll]) * self.sigmoid_prime(z_list[ll - 1])

        return (dW_list, db_list)

    def back_prop_dropout(self, x, y):
        """
        Back propagation with dropout on the hidden layers other than the output layer.

        Dropout layer can be thought of as a special linear layer between layers.
        """
        db_list = [np.zeros(b.shape) for b in self.biases]
        dW_list = [np.zeros(W.shape) for W in self.weights]

        a = x
        a_list = [a]
        z_list = [np.zeros(a.shape)]  # Pad with a placeholder so that indices match

        for index,(W, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(W,a)+b
            z_list.append(z)
            a = self.sigmoid(z)
            if index != len(self.weights)-1:
                for i in range(len(a)):
                    random_num = random.random()
                    if random_num > self.keep_prob:
                        a[i] = 0
            a_list.append(a)

        L = self.L
        delta = [np.zeros((n, 1)) for n in self.sizes]
        delta[L - 1] = self.grad_cost(a_list[L - 1], y) * self.sigmoid_prime(z_list[L - 1])

        for ll in range(L - 1, 0, -1):
            db_list[ll - 1] = delta[ll]
            dW_list[ll - 1] = np.dot(delta[ll], a_list[ll - 1].T)
            delta[ll - 1] = np.dot(self.weights[ll - 1].T, delta[ll]) * self.sigmoid_prime(z_list[ll - 1])

        return (dW_list, db_list)

    def evaluate(self, test_x,test_y):
        """
        Evaluate current model on labeled test data
        """
        ctr = 0
        for idx in range(len(test_x)):
            yhat = self.forward_prop(test_x[idx])
            ctr += np.argmax(yhat) == np.argmax(test_y[idx])
        return float(ctr) / float(len(test_x))

