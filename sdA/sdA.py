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
from DeepLearning.dA.dA import dA
from DeepLearning.MLP.mlp import FullyConnectedLayer
from DeepLearning.softmax.softmax import Softmax
from DeepLearning.common.training import supervised_tuning, relu
from theano.tensor.nnet import sigmoid

from DeepLearning.common.preProcessing import unit_scale

class sdA(object):
    """ Stacked denoising auto-encoder """

    """ The first facade of the sdA is a network of dAs. The Input to each dA is the 
    output code from the dA before. In the unsupervised pre-training phase, each 
    dA is trained seperately by attempting to minimise the error in reconstructing 
    its input (the output code from the layer before). The second facade is an MLP, 
    which over-lays the encoding parts of the sdA, and adds an final softmax 
    layer on top. In particular, the MLP hidden layers share the encoding weights of the 
    coresponding dAs that they over-lay, so that the optimal sdA parameters learned in 
    the pre-training phase will initialise those of the MLP. The latter is subsequently 
    trained in the supervsed fine-tuning phase. """
        

    def __init__(self, np_rng, dimIn, dimHiddenLayers, dimOut, dropout_rates, theano_rng=None):

        """ np_rng used for weights initialisations. theano_rng used to corrupt input 
        data."""

        self.X = T.matrix('X')
        self.y = T.ivector('y')
        
        self.nHiddenLayers = len(dimHiddenLayers)
        self.dA_layers = []
        self.FC_layers = [] # for the MLP hidden layers
        self.params = []

        assert self.nHiddenLayers > 0

        if not theano_rng:
            theano_rng = RandomStreams(123)

        # initialise the dA and FC layers

        for i in xrange(self.nHiddenLayers):
            if i == 0: # input layer 
                layerDimIn = dimIn
                layerInput = self.X
                if dropout_rates[0]!=0.0:
                    layerInputDropout = self.X
                else:
                    layerInputDropout = None

            else:
                layerDimIn = dimHiddenLayers[i-1]
                layerInput = self.FC_layers[-1].output # the FC output
                # doubles as the dA code
                try:
                    layerInputDropout = self.FC_layers[-1].output_dropout
                except:
                    if dropout_rates[i]!=0.0:
                        layerInputDropout=self.FC_layers[-1].output
                    else:
                        layerInputDropout = None

                
            # initialise the FC and corresponding dA for this layer.
            
            FC_layer = FullyConnectedLayer(np_rng,
                                           input = layerInput,
                                           dimIn = layerDimIn,
                                           dimOut = dimHiddenLayers[i],
                                           input_dropout=layerInputDropout,
                                           p_dropout=dropout_rates[i])
            
            dA_layer = dA(np_rng,
                          input = layerInput,
                          dimIn = layerDimIn,
                          dimHidden = dimHiddenLayers[i],
                          theano_rng=theano_rng,
                          W = FC_layer.W,
                          bEncode = FC_layer.b)

            
            self.FC_layers.append(FC_layer)
            self.dA_layers.append(dA_layer)
            
            self.params.extend(FC_layer.params) # store only the encoding params


        # add softmax on top of MLP

        softmaxDimIn = dimHiddenLayers[-1]
        softmaxInput = self.FC_layers[-1].output

        try:
            softmaxInputDropout = self.FC_layers[-1].output_dropout
        except:
            if dropout_rates[-1] != 0.0:
                softmaxInputDropout = self.FC_layers[-1].output
            else:
                softmaxInputDropout = None

        self.softmax_layer = Softmax(np_rng,
                                dimIn=softmaxDimIn,
                                dimOut=dimOut,
                                input=softmaxInput,
                                input_dropout=softmaxInputDropout,
                                p_dropout=dropout_rates[-1])
    
        
        self.params.extend(self.softmax_layer.params)

        self.cost_finetune = self.softmax_layer.cost
        self.error_finetune = self.softmax_layer.error


    def get_pretrain_fns(self, X_train, miniBatchSize, alpha_pre):

        corruption = T.scalar('corruption') # symbolic to allow variation between layers
        #in pre-training
        
        pretrain_fns = [] # holds pre-training functions for each dA layer
        
        for dA_layer in self.dA_layers:

            cost, updates = dA_layer.get_cost_updates(
                corruption,
                alpha_pre)

            idx = T.lscalar('idx') # minibatch index
            
            pretrain = theano.function([idx, corruption], cost, updates=updates,
                givens = {
                    self.X: X_train[idx*miniBatchSize: (idx+1)*miniBatchSize]
                }
            )
            
            pretrain_fns.append(pretrain)

        return pretrain_fns


    def get_finetune_functions(self, training_data, validation_data, test_data,
                               miniBatchSize, momentum=None):

        X_train, y_train = training_data
        X_val, y_val = validation_data
        X_test, y_test = test_data

        cost = self.cost_finetune(self.y) 
        grads = T.grad(cost, self.params)

        alpha = T.scalar('alpha') # Allows us to change during training
        
        if momentum is None:
            updates = [
                (layerParams, layerParams - alpha*layerGrads)
                for layerParams, layerGrads in zip(self.params, grads)
            ]

        else:
            assert (momentum >= 0 and momentum < 1)
            updates = []
            for layerParams, layerGrads in zip(self.params, grads):
                # Create shared variable to store parameter update.
                # Note this is only initialised once (on first iteration of SGD)
                momentum_update = theano.shared(layerParams.get_value()*0.,
                                             broadcastable=layerParams.broadcastable)
                # param_update refers to the updated param_update.
                updates.append((layerParams, layerParams - alpha*momentum_update))
                # Mix in previous (negative) step direction with current gradient.
                updates.append((momentum_update, momentum*momentum_update
                                + (1.-momentum)*layerGrads)) 
        
        idx = T.lscalar('idx') # minibatch index
        
        train_model = theano.function([idx, alpha], cost, updates=updates,
            givens = {
                self.X: X_train[idx*miniBatchSize: (idx+1)*miniBatchSize],
                self.y: y_train[idx*miniBatchSize: (idx+1)*miniBatchSize]
            }
        )
        
        
        test_model = theano.function([idx], self.error_finetune(self.y),
            givens = {
                self.X: X_test[idx * miniBatchSize: (idx + 1) * miniBatchSize],
                self.y: y_test[idx * miniBatchSize: (idx + 1) * miniBatchSize]
            }
        )
        

        validate_model = theano.function([idx], self.error_finetune(self.y),
            givens = {
                self.X: X_val[idx * miniBatchSize: (idx + 1) * miniBatchSize],
                self.y: y_val[idx * miniBatchSize: (idx + 1) * miniBatchSize]
            }
        )

        return train_model, validate_model, test_model


        

def apply_sdA_sgd(training_data, validation_data, test_data, nChannels,
                  dimHiddenLayers, nLabels, miniBatchSize, sgd_opts,
                  dropout_rates, corruptionLevels, results_dir,
                  momentum=None, monitoring_to_file=False):


    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    os.chdir(results_dir)

    dimIn = training_data[0].get_value(borrow=True).shape[1]
    nBatchTrain = training_data[0].get_value(borrow=True).shape[0]/miniBatchSize
    nBatchVal = validation_data[0].get_value(borrow=True).shape[0]/miniBatchSize
    nBatchTest = test_data[0].get_value(borrow=True).shape[0]/miniBatchSize

    np_rng = np.random.RandomState(0)
    theano_rng = RandomStreams(123)


    clf = sdA(np_rng,
              dimIn = training_data[0].get_value(borrow=True).shape[1],
              dimHiddenLayers=dimHiddenLayers,
              dimOut=nLabels,
              dropout_rates=dropout_rates)


    # compile pre-training functions
    
    pretrain_fns = clf.get_pretrain_fns(training_data[0], miniBatchSize, sgd_opts['alpha_pre'])
    

    # compile fine-tuning functions

    train_model, validate_model, test_model = clf.get_finetune_functions(
        training_data, validation_data, test_data, miniBatchSize, momentum)


    # do the actual pre-training

    print('starting pre-training of sdA...')


    start_time = timeit.default_timer()

    for layerIdx in xrange(clf.nHiddenLayers):
        for epoch in xrange(sgd_opts['epochs_pre']):
            monitorCost = 0.0
            for miniBatchIndex in xrange(nBatchTrain):
                iter = epoch*nBatchTrain+miniBatchIndex
                monitorCost += pretrain_fns[layerIdx](
                    miniBatchIndex,
                    corruptionLevels[layerIdx]) # aggregate miniBatch costs
                if (iter + 1) % 1000 == 0:
                    print('pre-training cost per minibatch for epoch {0}, minibatch {1}/{2}, layer {3}, is : {4:.5f}'
                      .format(epoch+1, miniBatchIndex+1, nBatchTrain, layerIdx+1, monitorCost/(iter+1)))

    end_time = timeit.default_timer()
    run_time_pre = end_time-start_time
    print 'pre-training time taken: {}'.format(run_time_pre)
    
    pretrain_params = [param.get_value(borrow=False) for param in clf.params]
    

    # do the actual fine-tuning
    print('starting fine-tuning of sdA...')

    [finetune_params, test_error, run_time, best_iter, temp_monitoring_filename] = supervised_tuning(
        clf.params, train_model, validate_model, test_model, nBatchTrain, nBatchVal,
        nBatchTest, sgd_opts, monitoring_to_file)

        # If device was set to gpu, check it was actually used

    if theano.config.device == 'gpu':
        if np.any([isinstance(x.op, cuda.GpuElemwise)
                   for x in
                   train_model.maker.fgraph.toposort()]):
            print 'used the gpu during training'
        else:
            print 'used the cpu during training'


    # save results

    trainingParams = {'sgd_opts':sgd_opts,
                      'miniBatchSize':miniBatchSize,
                      'dimIn': dimIn,
                      'dimHiddenLayers':dimHiddenLayers,
                      'dimOut': nLabels,
                      'dropout_rates':dropout_rates,
                      'corruptionLevels':corruptionLevels,
                      'momentum':momentum}

    results = ['test_error: {:.2%}'.format(test_error),
               'run_time: {:.2f}s'.format(run_time),
               'best_iter: {}'.format(best_iter)]

    with open('readme.txt', 'a') as f:
        f.write('\n {0}\n {1}\n'.format(trainingParams, results))
    f.close()

    if monitoring_to_file:
        monitoring_filename = 'sdA_{0:.2%}_{1}_monitoring.txt'.format(
            test_error, dimHiddenLayers)
        
        # rename temp file created during training
        os.rename(temp_monitoring_filename, monitoring_filename) 

        with open('readme.txt', 'a') as f:
            f.write('{0}\n'.format(monitoring_filename))
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
            imageFilename = 'sdA_filters_{0:.2%}_{1}.png'.format(
                test_error, dimHiddenLayers)
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
            modelFilename = 'sdA_{0:.2%}_{1}.pkl'.format(
                test_error, dimHiddenLayers)        
            best_model = {
                'params':finetune_params,
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



def test_saved_model(X_test, nLabels, filename, y_test=None):

    f = open(filename, 'rb')
    model = cPickle.load(f)
    saved_params = model['params']
    dimIn = model['dimIn']
    dimHiddenLayers = model['trainingParams']['dimHiddenLayers']
    dimOut = model['trainingParams']['dimOut']

    np_rng = np.random.RandomState(0)

    assert dimIn == X_test.get_value().shape[1]

    # build an sdA, but treat as an MLP (i.e. we won't be using the dA facade)
    clf = sdA(np_rng, dimIn, dimHiddenLayers, nLabels)

    for param, saved_param in zip(clf.params, saved_params):
        param.set_value(saved_param, borrow=True)
    

    idx = T.lscalar('idx')

    # for labels plot(s)
    zoomParams = {'zoom':30, 'x1':5100, 'x2':5200,
                  'y1':3100, 'y2':3200}
    
    if y_test is not None:
        assert X_test.get_value(borrow=True).shape[0]==y_test.eval().shape[0]
        assert y_test.eval().max() <= dimOut-1 and y_test.eval().min()>=0

        # reshape and plot ground truth

        print 'Plotting ground-truth'
        plot_labels_as_image(y_test.eval().reshape(imageShape),
                             cmap='terrain',
                             nColors=dimOut,
                             zoomParams=zoomParams)

        test_model = theano.function([idx], clf.error_finetune(clf.y),
            givens = {
                clf.X:X_test[idx*miniBatchSize: (idx+1)*miniBatchSize],
                clf.y:y_test[idx*miniBatchSize: (idx+1)*miniBatchSize]
            }
        )

        print 'Computing model error on test set...'
        test_error = np.mean([test_model(i) for i in xrange(nBatchTest)])
        
            
        print('test score {0:.2%}'
              .format(test_error)
        )


    # reshape and plot predictions

    get_predictions = theano.function([idx], clf.preds,
        givens = {
            clf.X:X_test[idx*miniBatchSize: (idx+1)*miniBatchSize]
        }
    )

    get_predictions_remainder = theano.function([], clf.preds,
        givens = {
            clf.X:X_test[nBatchTest*miniBatchSize: testShape[0]]
        }
    )

    print 'Generating predictions...'
    predictions = np.concatenate(tuple(get_predictions(i)
                                       for i in xrange(nBatchTest))
                                 +(get_predictions_remainder(),))

    print 'Plotting predictions...'
    plot_labels_as_image(predictions.reshape(imageShape),
                         cmap='terrain',
                         nColors=dimOut,
                         zoomParams=zoomParams)
        
        
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

    # unit scale if using cross entropy cost and not scaled already
    #print 'unit scaling data'
    #train_set[0], validation_set[0], test_set[0] = unit_scale([train_set[0],validation_set[0],test_set[0]])

    train_set, validation_set, test_set = [toShared_xy(train_set),
                                           toShared_xy(validation_set),
                                           toShared_xy(test_set)]    

    nChannels = 4
    corruptionLevels = [0.1, 0.2, 0.3]
    dropout_rates = [0.0,0.0,0.0,0.0]
    dimHiddenLayers = [100, 100, 100]
    nLabels = 4
    sgd_opts = {'epochs_pre':1, 'min_epochs':10, 'max_epochs':10,
                'alpha_pre':0.001, 'alpha_init':0.1, 'gamma':0.0001,
                'p':0.75, 'monitor_frequency':1000}
    momentum = 0.9
    miniBatchSize = 20
    monitoring_to_file=True
    
    if results_dir is not None:
        # train a new model
        print 'setting up model...'
        apply_sdA_sgd(
            train_set,
            validation_set,
            test_set,
            nChannels,
            dimHiddenLayers,
            nLabels,
            miniBatchSize,
            sgd_opts,
            dropout_rates,
            corruptionLevels,
            results_dir,
            momentum,
            monitoring_to_file)
        
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


