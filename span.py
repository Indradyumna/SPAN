import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class SPAN:
    """
    Implementation of Set Permutation Adversarial Networks (SPAN)
    """
    def __init__(self, n, dim, temp, lr, beta, numHidden, numLSTMUnits, numLSTMHidden, numLSTMOut):
        """
        Params:
            n: Number of elements in a set
            dim: Dimensionality of each element in set
            temp: Temperature parameter for Sinkhorn iterations
            lr: Learning Rate
            beta: L2 regularization loss coefficient
            numHidden: Number of hidden neurons in Permutation Adversarial Network
            numLSTMUnits: Number of hidden units in LSTM in learner
            numLSTMHidden: Number of hidden neurons in FC layer following LSTM in learner
            numLSTMOut: Output dimension
        """
        self.n = n
        self.dim = dim
        self.temp = temp
        self.lr = lr
        self.beta = beta
        self.numHidden = numHidden
        self.numLSTMUnits = numLSTMUnits
        self.numLSTMHidden = numLSTMHidden
        self.numLSTMOut = numLSTMOut
        
    def sinkhorn(self, X, num_iters=100):
        """
        Performs Sinkhorn iterations in the logarithmic scale on an input matrix to convert it into
        a doubly stochastic matrix - ref: https://github.com/google/gumbel_sinkhorn/blob/master/sinkhorn_ops.py
        Params:
            X: Tensor of shape (-1, n, n) on which Sinkhorn normalization is performed
            num_iters: Number of Sinkhorn iterations to be performed
        """
        assert X.get_shape().as_list()[1] == self.n
        assert X.get_shape().as_list()[2] == self.n
        X = tf.reshape(X, [-1, self.n, self.n])
        for _ in range(num_iters):
            X -= tf.reshape(tf.reduce_logsumexp(X, axis=2), [-1, self.n, 1])
            X -= tf.reshape(tf.reduce_logsumexp(X, axis=1), [-1, 1, self.n])
        return tf.exp(X)
    
    def createGraph(self):
        """
        Creates computation graph of SPAN
        """
        tf.reset_default_graph()
        
        self.X = tf.placeholder(tf.float32, [None, self.n, self.dim])
        self.Y = tf.placeholder(tf.float32, [None, self.numLSTMOut])
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.X_flattened = tf.reshape(self.X, [-1, self.dim])
        initializer = tf.contrib.layers.xavier_initializer()
        
        self.W1 = tf.Variable(initializer([self.dim, self.numHidden]))
        self.B1 = tf.Variable(initializer([self.numHidden]))
        self.hidden = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(self.X_flattened, self.W1), self.B1)), 
                                    keep_prob=self.keep_prob)
        
        self.W2 = tf.Variable(initializer([self.numHidden, self.n]))
        self.B2 = tf.Variable(initializer([self.n]))
        self.preSinkhorn = tf.add(tf.matmul(self.hidden, self.W2), self.B2)
        self.preSinkhorn = tf.reshape(self.preSinkhorn, [-1, self.n, self.n])
        self.preSinkhorn /= self.temp
        
        self.postSinkhorn = self.sinkhorn(self.preSinkhorn)

        self.postSinkhornInv = tf.transpose(self.postSinkhorn, [0, 2, 1])

        self.permutedX = tf.matmul(self.postSinkhornInv, self.X)
        
        self.LSTMW1 = tf.Variable(initializer([self.numLSTMUnits, self.numLSTMHidden]))
        self.LSTMB1 = tf.Variable(initializer([self.numLSTMHidden]))

        self.LSTMW2 = tf.Variable(initializer([self.numLSTMHidden, self.numLSTMOut]))
        self.LSTMB2 = tf.Variable(initializer([self.numLSTMOut]))

        self.permutedXUnstacked = tf.unstack(self.permutedX, self.n, axis=1)
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.numLSTMUnits, forget_bias=1)
        self.outputs, self.states = tf.nn.static_rnn(self.cell, self.permutedXUnstacked, dtype=tf.float32)
        self.LSTMHidden = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(self.outputs[-1], self.LSTMW1), self.LSTMB1)),
                                        keep_prob=self.keep_prob)
        self.LSTMOut = tf.add(tf.matmul(self.LSTMHidden, self.LSTMW2), self.LSTMB2)

        self.loss = tf.nn.l2_loss(self.LSTMOut - self.Y)

        self.varListFunction = [self.LSTMW1, self.LSTMB1, self.LSTMW2, self.LSTMB2]
        self.varListFunction.extend(self.cell.variables)
        self.varListPerm = [self.W1, self.B1, self.W2, self.B2]

        self.regLossFunction = self.beta * tf.add_n([tf.nn.l2_loss(var) for var in self.varListFunction])

        self.optimizerFunction = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.loss + self.regLossFunction, var_list=self.varListFunction)
        self.optimizerPerm = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            -1 * self.loss, var_list=self.varListPerm)
        
    def updateSavedWeights(self):
        """
        Updates saved weights to present values of Variables
        """
        assert self.sess is not None
        self.weights = {
            'W1': self.sess.run(self.W1),
            'B1': self.sess.run(self.B1),
            'W2': self.sess.run(self.W2),
            'B2': self.sess.run(self.B2),
            'LSTMW1': self.sess.run(self.LSTMW1),
            'LSTMB1': self.sess.run(self.LSTMB1),
            'LSTMW2': self.sess.run(self.LSTMW2),
            'LSTMB2': self.sess.run(self.LSTMB2),
            'kernel': self.sess.run(self.cell.variables[0]),
            'bias': self.sess.run(self.cell.variables[1])
        }
        
    def getWeights(self):
        """
        Returns saved weights
        """
        return self.weights
    
    def setWeights(self, weights):
        """
        Sets weights field to the one provided
        """
        self.weights = weights
    
    def assignWeights(self, weights):
        """
        Assigns provided weights to the Variables
        """
        if self.sess is None:
            self.sess = tf.Session()
        self.sess.run(tf.assign(self.W1, weights['W1']))
        self.sess.run(tf.assign(self.B1, weights['B1']))
        self.sess.run(tf.assign(self.W2, weights['W2']))
        self.sess.run(tf.assign(self.B2, weights['B2']))
        self.sess.run(tf.assign(self.LSTMW1, weights['LSTMW1']))
        self.sess.run(tf.assign(self.LSTMB1, weights['LSTMB1']))
        self.sess.run(tf.assign(self.LSTMW2, weights['LSTMW2']))
        self.sess.run(tf.assign(self.LSTMB2, weights['LSTMB2']))
        self.sess.run(tf.assign(self.cell.variables[0], weights['kernel']))
        self.sess.run(tf.assign(self.cell.variables[1], weights['bias']))
          
    def createSessionAndInit(self):
        """
        Creates new session and initiliazes Variables prior to training
        """
        self.sess = tf.Session()
        self.trainLossList = []
        self.minValLoss = 10000000000000
        self.sess.run(tf.global_variables_initializer())
        self.updateSavedWeights()
        
    def train(self, x_train, y_train, x_val, y_val, numEpochs, numSubEpochsPerm, 
              numSubEpochsFunction, batchSize, keep_prob=1.0, boost=False, boostEpochs=0):
        """
        Alternating block coordinate descent to train SPAN to learn a permutation-invariant set function
        Params:
            x_train: Training data - Numpy array of shape (-1, n, dim)
            y_train: Training labels - Numpy array of shape (-1, 1)
            x_val: Validation data - Numpy array of shape (-1, n, dim)
            y_val: Validation labels - Numpy array of shape (-1, 1)
            numEpochs: Number of epochs of alternating block coordinate descent to perform
            numSubEpochsPerm: Number of epochs to adversarially train Permutation Adversarial Network
                              by maximizing loss
            numSubEpochsFunction: Number of epochs to train learner by minimizing loss
            batchSize: Batch size
            keep_prob: Dropout parameter
            boost: Optional flag to run additional iterations to optimize learner
            boosEpochs: Number of epochs of boosting to run if boost is set to True
        """
        for i in range(numEpochs):
            for j in range(numSubEpochsPerm):
                totalLoss = 0
                for k in range(0, len(x_train), batchSize):
                    batch_x_train = x_train[k:min(k+batchSize, len(x_train)), :, :]
                    batch_y_train = y_train[k:min(k+batchSize, len(y_train)), :]
                    _, l = self.sess.run([self.optimizerPerm, self.loss],
                                         feed_dict={self.X:batch_x_train, self.Y:batch_y_train, 
                                                    self.keep_prob:keep_prob})
                    totalLoss += l
                self.trainLossList.append(totalLoss)
                print("Optimizing Permutation: Epoch %d Sub-epoch %d Loss %f" % (i, j, totalLoss), end='\r')
            print()
            for j in range(numSubEpochsFunction):
                totalLoss = 0
                for k in range(0, len(x_train), batchSize):
                    batch_x_train = x_train[k:min(k+batchSize, len(x_train)), :, :]
                    batch_y_train = y_train[k:min(k+batchSize, len(y_train)), :]
                    _, l = self.sess.run([self.optimizerFunction, self.loss],
                                         feed_dict={self.X:batch_x_train, self.Y:batch_y_train,
                                                    self.keep_prob:keep_prob})
                    totalLoss += l
                self.trainLossList.append(totalLoss)
                print("Optimizing Function: Epoch %d Sub-epoch %d Loss %f" % (i, j, l), end='\r')
            print()
            valLoss = self.sess.run(self.loss, feed_dict={self.X:x_val, self.Y:y_val, self.keep_prob:1.0})
            print("Validation loss is %f" % valLoss)
            if valLoss < self.minValLoss:
                self.minValLoss = valLoss
                self.updateSavedWeights()
        if boost:
            for j in range(boostEpochs):
                totalLoss = 0
                for k in range(0, len(x_train), batchSize):
                    batch_x_train = x_train[k:min(k+batchSize, len(x_train)), :, :]
                    batch_y_train = y_train[k:min(k+batchSize, len(y_train)), :]
                    _, l = self.sess.run([self.optimizerFunction, self.loss],
                                         feed_dict={self.X:batch_x_train, self.Y:batch_y_train,
                                                    self.keep_prob:keep_prob})
                    totalLoss += l
                self.trainLossList.append(totalLoss)
                print("Boosting - Optimizing Function: Epoch %d Sub-epoch %d Loss %f" % (i, j, l), end='\r')
                valLoss = self.sess.run(self.loss, feed_dict={self.X:x_val, self.Y:y_val, self.keep_prob:1.0})
                print("Validation loss is %f" % valLoss)
                if valLoss < self.minValLoss:
                    self.minValLoss = valLoss
                    self.updateSavedWeights()
        
        return self.getWeights()
    
    def predict(self, x):
        """
        Return SPAN predictions on input
        Params:
            x: Input to generate predictions on
        """
        preds = self.sess.run(self.LSTMOut, feed_dict={self.X:x, self.keep_prob:1.0})
        return preds
    
    def plotTrainingLoss(self):
        """
        Plots training loss over training epochs
        """
        plt.plot(self.trainLossList)
        plt.show()
        
    def exportWeightsAsNumpy(self, directory):
        """
        Dumps weights as numpy arrays to directory
        Params:
            Directory to store weights in
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + "/W1", weights['W1'])
        np.save(directory + "/B1", weights['B1'])
        np.save(directory + "/W2", weights['W2'])
        np.save(directory + "/B2", weights['B2'])
        np.save(directory + "/LSTMW1", weights['LSTMW1'])
        np.save(directory + "/LSTMB1", weights['LSTMB1'])
        np.save(directory + "/LSTMW2", weights['LSTMW2'])
        np.save(directory + "/LSTMB2", weights['LSTMB2'])
        np.save(directory + "/kernel", weights['kernel'])
        np.save(directory + "/bias", weights['bias'])
        
    def importWeightsAsNumpy(self, directory):
        """
        Imports and returns weights from numpy arrays stored in directory
        Params:
            directory: Directory containing saved weights as numpy arrays
        """
        weights = {
           'W1': "",
           'B1': "",
           'W2': "",
           'B2': "",
           'LSTMW1': "",
           'LSTMB1': "",
           'LSTMW2': "",
           'LSTMB2': "",
           'kernel': "",
           'bias': ""
        }
        for key, value in weights.items():
            arr = np.load(directory + "/%s.npy" % key)
            weights[key] = arr
        
        return weights
    
    def exportTFGraph(self):
        pass
    
    def importTFGraph(self):
        pass
