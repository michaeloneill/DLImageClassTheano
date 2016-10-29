import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

import os
import sys, getopt
from PIL import Image
import cPickle

import timeit
from DeepLearning.common.loadData import load_data_pickled, load_data_npz, toShared_xy
from DeepLearning.common.plotting import tile_raster_images
from DeepLearning.rbm.rbm import rbm
from DeepLearning.MLP.mlp import FullyConnectedLayer
from DeepLearning.softmax.softmax import Softmax
from DeepLearning.common.training import supervised_tuning



class dbn(object):
    """ Deep belief network """

    """ The first facade of the dbn is a network of rbms. The Input to each rbm is the 
    hidden representation (samples or mean activations) from the rbm before. In the 
    unsupervised pre-training phase, each rbm is trained seperately by attempting to 
    minimise the error in reconstructing its input (using a proxy cost). The second 
    facade is an MLP, which over-lays the hidden parts of the dbn,and adds an final 
    softmax layer on top. In particular, the MLP hidden layers share the weights and 
    hidden layer biases coresponding rbms that they over-lay, so that the optimal rbm 
    parameters learned in the pre-training phase will initialise those of the MLP. The 
    latter is subsequently trained in the supervsed fine-tuning phase. 
    """
        

    
    def __init__(self, np_rng, dimIn, dimHiddenLayers, dimOut, theano_rng=None):

        """ np_rng used for weights initialisations. """

        self.X = T.matrix('X')
        self.y = T.ivector('y')
        
        self.nHiddenLayers = len(dimHiddenLayers)
        self.rbm_layers = []
        self.sigmoid_layers = [] # for the MLP hidden layers
        self.params = []

        assert self.nHiddenLayers > 0

        if not theano_rng:
            theano_rng = RandomStreams(123)

        # initialise the rbm and MLP layers

        for i in xrange(self.nHiddenLayers):
            if i == 0: # input layer 
                layerDimIn = dimIn
                layerInput = self.X
            else:
                layerDimIn = dimHiddenLayers[i-1]
                layerInput = self.sigmoid_layers[-1].output # the sigmoid output
                # doubles as the rbm code
                
            # initialise the sigmoid and corresponding rbm for this layer.
            
            sigmoid_layer = FullyConnectedLayer(np_rng,
                                         input = layerInput,
                                         dimIn = layerDimIn,
                                         dimOut = dimHiddenLayers[i])
            
            rbm_layer = rbm(np_rng,
                          input = layerInput,
                          dimIn = layerDimIn,
                          dimHidden = dimHiddenLayers[i],
                          theano_rng=theano_rng,
                          W = sigmoid_layer.W,
                          b_h = sigmoid_layer.b)

            
            self.sigmoid_layers.append(sigmoid_layer)
            self.rbm_layers.append(rbm_layer)
            
            self.params.extend(sigmoid_layer.params) # store only the encoding params


        # add softmax on top of MLP
        
        self.softmax_layer = Softmax(np_rng, input=self.sigmoid_layers[-1].output,
                                     dimIn=dimHiddenLayers[-1],
                                     dimOut=dimOut)
        
        self.params.extend(self.softmax_layer.params)

        self.cost_finetune = self.softmax_layer.cost(self.y)
        self.error_finetune = self.softmax_layer.error(self.y)

    def get_pretrain_fns(self, X_train, miniBatchSize, alpha_pre, persistent_bool, k):

        pretrain_fns = [] # holds pre-training functions for each rbm layer

        
        for rbm_layer in self.rbm_layers:
            
            # whether to use CD-k or PCD-k. Must create seperately for each layer
            if persistent_bool:
                persistent_chain = theano.shared(
                    np.zeros(
                        (miniBatchSize, rbm_layer.dimHidden),
                        dtype=theano.config.floatX),
                    borrow=True)
            else:
                persistent_chain = None
            
            cost, updates = rbm_layer.get_cost_updates(
                alpha_pre, persistent=persistent_chain, k=k)

            idx = T.lscalar('idx') # minibatch index
            
            pretrain = theano.function([idx], cost, updates=updates,
                givens = {
                    self.X: X_train[idx*miniBatchSize: (idx+1)*miniBatchSize]
                }
            )
            
            pretrain_fns.append(pretrain)

        return pretrain_fns


    def get_finetune_functions(self, training_data, validation_data, test_data,
                               miniBatchSize):

        X_train, y_train = training_data
        X_val, y_val = validation_data
        X_test, y_test = test_data

        cost = self.cost_finetune 
        grads = T.grad(cost, self.params)

        alpha = T.scalar('alpha') # Allows us to change during training
        
        updates = [
            (layerParams, layerParams-alpha*layerGrads)
            for layerParams, layerGrads in zip(self.params, grads)
        ]
        
        idx = T.lscalar('idx') # minibatch index
        
        train_model = theano.function([idx, alpha], cost, updates=updates,
            givens = {
                self.X: X_train[idx*miniBatchSize: (idx+1)*miniBatchSize],
                self.y: y_train[idx*miniBatchSize: (idx+1)*miniBatchSize]
            }
        )
        
        
        test_model = theano.function([idx], self.error_finetune,
            givens = {
                self.X: X_test[idx * miniBatchSize: (idx + 1) * miniBatchSize],
                self.y: y_test[idx * miniBatchSize: (idx + 1) * miniBatchSize]
            }
        )
        

        validate_model = theano.function([idx], self.error_finetune,
            givens = {
                self.X: X_val[idx * miniBatchSize: (idx + 1) * miniBatchSize],
                self.y: y_val[idx * miniBatchSize: (idx + 1) * miniBatchSize]
            }
        )

        return train_model, validate_model, test_model


def apply_dbn_sgd(training_data, validation_data, test_data, nChannels, dimHiddenLayers, nLabels, miniBatchSize, sgd_opts, persistent_bool, k, results_dir):

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    os.chdir(results_dir)

    dimIn = training_data[0].get_value(borrow=True).shape[1]
    nBatchTrain = training_data[0].get_value(borrow=True).shape[0]/miniBatchSize
    nBatchVal = validation_data[0].get_value(borrow=True).shape[0]/miniBatchSize
    nBatchTest = test_data[0].get_value(borrow=True).shape[0]/miniBatchSize
        
    np_rng = np.random.RandomState(0)
    theano_rng = RandomStreams(123)


    clf = dbn(np_rng,
              dimIn = training_data[0].get_value(borrow=True).shape[1],
              dimHiddenLayers=dimHiddenLayers,
              dimOut=nLabels)


    # get pre-training functions
    
    pretrain_fns = clf.get_pretrain_fns(training_data[0], miniBatchSize, sgd_opts['alpha_pre'], persistent_bool, k) 



    # get fine-tuning functions
    
    train_model, validate_model, test_model = clf.get_finetune_functions(
        training_data, validation_data, test_data, miniBatchSize)
    

    # do the actual pre-training

    print('starting pre-training of deep belief...')

    start_time = timeit.default_timer()

    for layerIdx in xrange(clf.nHiddenLayers):
        for epoch in xrange(sgd_opts['epochs_pre']):
            monitorCost = 0.0
            for miniBatchIndex in xrange(nBatchTrain):
                iter = epoch*nBatchTrain+miniBatchIndex
                monitorCost += pretrain_fns[layerIdx](
                    miniBatchIndex) # aggregate miniBatch costs
            
                if (iter + 1) % 1000 == 0:
                    print('pre-training cost per minibatch for epoch {0}, minibatch {1}/{2}, layer {3}, is : {4:.5f}'
                    .format(epoch+1, miniBatchIndex+1, nBatchTrain, layerIdx+1, monitorCost/(iter+1)))

    end_time = timeit.default_timer()
    run_time_pre = end_time-start_time
    print 'pre-training time taken: {}'.format(run_time_pre)


    pretrain_params = [param.get_value(borrow=False) for param in clf.params]

    
    # do the actual fine-tuning
    print('starting fine-tuning of deep belief...')

    [finetune_params, test_error, run_time, best_iter, _] = supervised_tuning(
        clf.params, train_model, validate_model, test_model, nBatchTrain,
        nBatchVal, nBatchTest, sgd_opts)

    # save results

    trainingParams = {'sgd_opts':sdg_opts,
                      'miniBatchSize':miniBatchSize,
                      'dimHiddenLayers':dimHiddenLayers,
                      'persistent_bool':persistent_bool, 'k':k}

    results = ['test_error: {:.2%}'.format(test_error),
               'run_time: {:.2f}s'.format(run_time),
               'best_iter: {}'.format(best_iter)]

    with open('readme.txt', 'a') as f:
        f.write('\n {0}\n {1}\n'.format(trainingParams, results))
    f.close()


    while True:

        answer = raw_input('plot pre-train filters (y/n)?')

        if answer == 'y':

            # saving pre-training filters

            if nChannels>1:
                dimImage = int(np.sqrt(dimIn/nChannels)) # assume square image
                tile_shape = (10, nChannels) # plot first 10 filters across each channel
                X=pretrain_params[0].T.reshape(-1, dimImage*dimImage) # first nChannels rows relate to first filter (adjacent rows seperate channels)
            else:
                dimImage = int(np.sqrt(dimIn)) # assume square image
                tile_shape = (10, 10) # or just plot first 100 filters
                X=pretrain_params[0].T


            imageFilename = 'dbn_filters_{0}_{1:.2%}_persistence={2}_k={3}.png'.format(
                dimHiddenLayers, test_error, persistent_bool, k)
            image = Image.fromarray(tile_raster_images(
                X=X,
                img_shape=(dimImage, dimImage), tile_shape=tile_shape,
                tile_spacing=(1, 1)))
            image.save(imageFilename)

            
            with open('readme.txt', 'a') as f:
                f.write('{0}\n'.format(imageFilename))
            f.close()

            break

        elif answer == 'n':

            break

        else:

            print('invalid input, try again')


    while True:
            
        answer = raw_input('save model (y/n)? ')

        if answer == 'y':
            
            modelFilename = 'dbn_{0:.2%}_{1}_persistence={2}_k={3}.pkl'.format(
                test_error, dimHiddenLayers, persistent_bool, k)

            best_model = {
                'dimIn': dimIn, 
                'params':finetune_params,
                'results':['test_error: {:.2%}'.format(test_error),
                           'run_time: {:.2f}s'.format(run_time)],
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
    dimHiddenLayers = model['trainingParams']['dimHiddenLayers']


    X_test, y_test = test_data
    np_rng = np.random.RandomState(0)

    assert dimIn == X_test.get_value().shape[1], 'dimensions of test data incompatible with saved model'

    # build a dbn, but treat as an MLP (i.e. we won't be using the rbm facade)
    clf = dbn(np_rng, dimIn, dimHiddenLayers, nLabels)

    for param, saved_param in zip(clf.params, saved_params):
        param.set_value(saved_param, borrow=True)
    
    test_model = theano.function([], clf.error_finetune,
        givens = {
            clf.X:X_test,
            clf.y:y_test
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
        opts, args = getopt.getopt(argv, "hd:r:m:", ["data_file=", "results_dir=", "model_file="])
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

    try:
        print 'loading in data...'
        train_set, validation_set, test_set = load_data_npz(data_file)
    except IOError:
        print 'cannot open data_file', data_file

    train_set, validation_set, test_set = [toShared_xy(train_set),
                                           toShared_xy(validation_set),
                                           toShared_xy(test_set)]    
    nChannels = 4
    dimHiddenLayers = [1000, 1000, 1000]
    nLabels = 4
    sgd_opts = {'epochs_pre':15, 'min_epochs':15, 'max_epochs':15,
                'alpha_pre':0.001, 'alpha_init':0.1, 'gamma':0.0001,
                'p':0.75}
    miniBatchSize = 20
    persistent_bool = True
    k = 1
    
    if results_dir is not None:
        # train a new model
        print 'setting up model...'
        apply_dbn_sgd(
            train_set, validation_set, test_set, nChannels, dimHiddenLayers,
            nLabels, miniBatchSize, sgd_opts, persistent_bool, k, results_dir)
    elif model_file is not None:
        # test a saved model
        # this option will be ignored if results_dir is not None
        print 'testing saved model...'
        test_saved_model(test_set, nLabels, model_file) 
    else:
        print 'exactly 2 arguments must be passed on command line'
        print 'usage1: softmax.py -d <data_file> -r <results_dir>'
        print 'usage2: softmax.py -d <data_file> -m <saved_model_file>'


if __name__ == "__main__":

    main(sys.argv[1:])





    








                      
