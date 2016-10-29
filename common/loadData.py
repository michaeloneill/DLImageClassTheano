import theano
import theano.tensor as T
import numpy as np

import cPickle
import scipy.io


def load_data_pickled(filename, only_test=False):

    f = open(filename, 'rb')
    if only_test:
        test_set = cPickle.load(f)
        f.close()
        return test_set
    else:
        train_set, validation_set, test_set = cPickle.load(f)
        f.close()
        return [train_set, validation_set, test_set]

def load_data_npz(filename, only_test=False):

    f = open(filename, 'rb')
    data = np.load(filename)
    f.close()

    if only_test:
        test_set = [data['X_test'], data['y_test']]
        return test_set
    else:
        train_set = [data['X_train'], data['y_train']]
        validation_set = [data['X_val'], data['y_val']]
        test_set = [data['X_test'], data['y_test']]
        return [train_set, validation_set, test_set]

def toShared_xy(data_xy):
    
    shared_X = theano.shared(
        np.asarray(data_xy[0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
        np.asarray(data_xy[1], dtype=theano.config.floatX), borrow=True)
    # work with an int32 cast of y instead of the floatX shared version
    return shared_X, T.cast(shared_y, "int32")


def toShared_x(data_x):
    
    shared_X = theano.shared(
        np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    return shared_X



    
