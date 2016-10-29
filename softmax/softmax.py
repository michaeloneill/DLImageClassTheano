import theano
import theano.tensor as T
import numpy as np
import cPickle
import timeit
import os
import sys, getopt

from theano.sandbox import cuda
from DeepLearning.common.loadData import load_data_pickled, load_data_npz, toShared_xy
from DeepLearning.common.training import supervised_tuning, dropout_from_layer

class Softmax(object):

    def __init__(self, rng, dimIn, dimOut,
                 input=None, input_dropout=None, p_dropout=0.0):

        if p_dropout != 0.0:
            assert input_dropout is not None, 'To apply dropout'
            ' input_dropout must be passed.'

        self.dimIn = dimIn
        self.dimOut = dimOut
        self.p_dropout = p_dropout
        
        epsilon = np.sqrt(6./(dimIn+dimOut)) # from Xavier 2010 (softmax)
        
        W_values = np.asarray(
            rng.uniform(
                low = -epsilon,
                high = epsilon,
                size=(self.dimIn, self.dimOut)),
            dtype=theano.config.floatX)

        b_values = np.zeros(dimOut, dtype=theano.config.floatX)

        self.W = theano.shared(value = W_values, name = 'W', borrow = True)
        self.b = theano.shared(value = b_values, name = 'b', borrow = True)
        self.params = [self.W, self.b]

        self.L1 = (
            abs(self.W).sum()
        )

        self.L2_sqr = (
            (self.W ** 2).sum()
        )
        
        # Set symbolic inputs and outputs.

        if input is not None:
            self.input = input
            # Scale weights by (1-p_dropout).
            self.output = T.nnet.softmax(
                (1-self.p_dropout)*T.dot(
                    self.input, self.W) + self.b) # batchSize by nLabels
            self.preds = T.argmax(self.output, axis=1)
        
        if input_dropout is not None:
            self.input_dropout = dropout_from_layer(input_dropout, p_dropout)
            self.output_dropout = T.nnet.softmax(T.dot(self.input_dropout, self.W)
                                                 + self.b)
        else:
            self.output_dropout = self.output

    def cost(self, y):
        
        # [T.arange(y.shape[0]), y] are the numPatches indices to pick out
        # from the log probability matrix. The corresponding values are then
        # averaged over. 
        # The cost we want to minimise is that involving output_dropout.
        # This is equal to self.output in the case no dropout was performed.
        return -T.mean(T.log(self.output_dropout)[T.arange(y.shape[0]), y]) 

    def error(self, y):
        
        return T.mean(T.neq(y, self.preds))

    def get_training_functions(self, X, y, training_data, validation_data,
                               test_data, miniBatchSize, lmbda):

        X_train, y_train = training_data
        X_val, y_val = validation_data
        X_test, y_test = test_data

        cost = self.cost(y) + lmbda*self.L2_sqr
        grads = T.grad(cost, self.params)

        alpha = T.scalar('alpha') # Allows us to change during training

        updates = [
            (param, param - alpha*grad)
            for param, grad in zip(self.params, grads)
        ]

        idx = T.lscalar('idx') # minibatch index
    
        # Compile function to train model.
        train_model = theano.function([idx, alpha], cost, updates=updates,
            givens = {
                X: X_train[idx*miniBatchSize: (idx+1)*miniBatchSize],
                y: y_train[idx*miniBatchSize: (idx+1)*miniBatchSize],
            }
        )

        # Compile function to compute error on test set.
        test_model = theano.function([idx], self.error(y),
            givens = {
                X: X_test[idx * miniBatchSize: (idx + 1) * miniBatchSize],
                y: y_test[idx * miniBatchSize: (idx + 1) * miniBatchSize]
            }
        )

        # Compile function to compute error on validation set.
        validate_model = theano.function([idx], self.error(y),
            givens = {
                X: X_val[idx * miniBatchSize: (idx + 1) * miniBatchSize],
                y: y_val[idx * miniBatchSize: (idx + 1) * miniBatchSize]
            }
        )

        
        return train_model, validate_model, test_model

        
def apply_softmax_sgd(training_data, validation_data, test_data,
                      nLabels, miniBatchSize, sgd_opts, lmbda, results_dir,
                      monitoring_to_file=False):
        
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    os.chdir(results_dir)
    
    X = T.matrix('X')
    y = T.ivector('y')

    X_train, y_train = training_data
    X_val, y_val = validation_data
    X_test, y_test = test_data
    
    nBatchTrain = X_train.get_value(borrow=True).shape[0]/miniBatchSize
    nBatchVal = X_val.get_value(borrow=True).shape[0]/miniBatchSize
    nBatchTest = X_test.get_value(borrow=True).shape[0]/miniBatchSize

    rng = np.random.RandomState(0)
    dimIn = X_train.get_value(borrow=True).shape[1]
    clf = Softmax(rng = rng, input = X, dimIn = dimIn, dimOut = nLabels)

    train_model, validate_model, test_model = clf.get_training_functions(
        X,
        y,
        training_data,
        validation_data,
        test_data,
        miniBatchSize,
        lmbda)


    # Do the actual training.

    print('starting training...')
        
    [best_params, test_error, run_time, best_iter, temp_monitoring_filename] = supervised_tuning(
        clf.params, train_model, validate_model, test_model,
        nBatchTrain, nBatchVal, nBatchTest, sgd_opts
    ) 
        

    # check if GPU was used
    
    if np.any([isinstance(x.op, cuda.GpuElemwise)
               for x in
               train_model.maker.fgraph.toposort()]):
        print 'used the gpu during training'
    else:
        print 'used the cpu during training'


    # Save results

    trainingParams = {'sgd_opts':sgd_opts, 'miniBatchSize':miniBatchSize,
                      'lmbda':lmbda}
    results = ['test_error: {:.2%}'.format(test_error),
               'run_time: {:.2f}s'.format(run_time),
               'best_iter: {}'.format(best_iter)]
    
    with open('readme.txt', 'a') as f:
        f.write('\n {0}\n {1}\n'.format(trainingParams, results))
    f.close()


    if monitoring_to_file:
        monitoring_filename = 'softmax_{0:.2%}_monitoring.txt'.format(
            test_error)
        
        # rename temp file created during training
        os.rename(temp_monitoring_filename, monitoring_filename) 

        with open('readme.txt', 'a') as f:
            f.write('{0}\n'.format(monitoring_filename))
        f.close()

    
    while True:
            
        answer = raw_input('save model (y/n)? ')

        if answer == 'y':
            modelFilename = 'softmax_{0:.2%}.pkl'.format(test_error)
            best_model = {
                'dimIn': dimIn,
                'params':best_params,
                'results': results,
                'trainingParams':trainingParams}
            
            with open(modelFilename, 'wb') as f:
                cPickle.dump(best_model, f, -1)
            f.close()

            with open('readme.txt', 'a') as f:
                f.write('{0}\n'.format(modelFilename))
            f.close()
            break

        elif answer =='n':
            break
        else:
            print('invalid input, try again')
    
    os.chdir('../')



def test_saved_model(test_data, nLabels, filename):

    f = open(filename, 'rb')
    model = cPickle.load(f)
    saved_params = model['params']
    dimIn = model['dimIn']
    
    X = T.matrix('X')
    y = T.ivector('y')

    X_test, y_test = test_data
    rng = np.random.RandomState(0)

    assert dimIn == X_test.get_value(borrow=True).shape[1], 'Dimensions of'
    ' test data incompatible with saved model.'
    
    clf = Softmax(rng = rng, input = X, dimIn = dimIn, dimOut = nLabels)

    for param, saved_param in zip(clf.params, saved_params):
        param.set_value(saved_param, borrow=True)
    
    test_model = theano.function([], clf.error(y),
        givens = {
            X:X_test,
            y:y_test
        }
    )

    test_error = float(test_model())
    print('test score {0:.2%}'
          .format(test_error)
    )

    
def main(argv):

    data_file = None
    results_dir = None
    model_file = None

    try:
        opts, args = getopt.getopt(
            argv, "hd:r:m:",
            ["data_file=", "results_dir=", "model_file="])
        
    except getopt.GetoptError:
        print 'incorrect usage'
        print 'usage1: softmax.py -d <data_file> -r <results_dir>'
        print 'usage2: softmax.py -d <data_file> -m <saved_model_file>'
        sys.exit(2)
        
    for opt, arg in opts:
        if opt=="-h":
            print 'usage1: softmax.py -d <data_file> -r <results_dir>'
            print 'usage2: softmax.py -d <data_file> -m <saved_model_file>'
        elif opt in ("-d", "--data_file"):
            data_file = arg
        elif opt in ("-r", "--results__dir"):
            results_dir = arg
        elif opt in ("-m", "--model_file"):
            model_file = arg
            
    if data_file is None:
        print 'data_file was not given'
        print 'usage1: softmax.py -d <data_file> -r <results_dir>'
        print 'usage2: softmax.py -d <data_file> -m <saved_model_file>'
        sys.exit(2)

    if results_dir is not None:

        # Train a new model.
    
        nLabels=10
        sgd_opts = {'min_epochs':10, 'max_epochs':10, 'alpha_init':0.1, 
                'gamma':0.0001, 'p':0.75}
        miniBatchSize = 600
        lmbda = 0.0

        try:
            train_set, validation_set, test_set = load_data_pickled(data_file)
        
        except IOError:
            print 'cannot open data_file', data_file

        train_set, validation_set, test_set = [toShared_xy(train_set),
                                               toShared_xy(validation_set),
                                               toShared_xy(test_set)]    
        apply_softmax_sgd(
            train_set, validation_set, test_set, nLabels,
            miniBatchSize, sgd_opts, lmbda, results_dir)
        
    elif model_file is not None:

        # Test a saved model.

        nLabels = 4
        try:
            _, _, test_set = load_data_npz(data_file)
        except IOError:
            print 'cannot open data_file', data_file

        test_set = toShared_xy(test_set)    
        test_saved_model(test_set, nLabels, model_file)
        
    else:
        print 'exactly 2 arguments must be passed on command line'
        print 'usage1: softmax.py -d <data_file> -r <results_dir>'
        print 'usage2: softmax.py -d <data_file> -m <saved_model_file>'


if __name__ == "__main__":

    main(sys.argv[1:])

    

    
    


    
    
                            

        
