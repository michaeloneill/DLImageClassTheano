import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from theano.tensor.nnet import softmax, sigmoid
from theano.tensor import tanh

from sklearn.metrics import confusion_matrix

import numpy as np
import os
import sys, getopt
from PIL import Image
import cPickle
import timeit

from theano.sandbox import cuda
from DeepLearning.softmax.softmax import Softmax
from DeepLearning.MLP.mlp import FullyConnectedLayer, dropout_from_layer, relu
from DeepLearning.common.loadData import load_data_pickled, load_data_npz, toShared_xy, toShared_x
from DeepLearning.common.training import supervised_tuning, dropout_from_layer, relu
from DeepLearning.common.plotting import tile_raster_images, plot_predictions, plot_labels_as_image, plot_confusion_matrix


class cnn(object):

    def __init__(self, rng, layers, mc_samples=None):
        
        self.layers = layers
        self.params = [param for layer in self.layers
                       for param in layer.params]
        self.cost = self.layers[-1].cost # function pointer
        
        if mc_samples is None:
            # Standard dropout network.
            try:
                self.preds = self.layers[-1].preds
                self.error = self.layers[-1].error # function pointer
            except:
                print('Could not access network outputs'
                      ' - did you pass a (non-dropout) input?'
                      )
        else:
            # mc_dropout network.
            self.mc_samples = mc_samples 
            mc_outputs, _ = theano.scan(lambda: self.layers[-1].output_dropout,
                                        outputs_info=None,
                                        n_steps = self.mc_samples)
            
            self.predictive_distribution_mean = T.mean(mc_outputs, axis=0)
            self.predictive_distribution_var = T.var(mc_outputs, axis=0)
            self.preds = T.argmax(self.predictive_distribution_mean, axis=1)
            self.error = self.__error_mc

        self.L1 = (
            T.sum([abs(layer.W).sum() for layer in self.layers]) 
        )
        self.L2_sqr = (
            T.sum([(layer.W ** 2).sum() for layer in self.layers])
        )

    def __error_mc(self, y): 
        return T.mean(T.neq(y, self.preds)) # will call mc_dropout preds
        
    def get_training_functions(self, X, y, training_data, validation_data,
                               test_data, miniBatchSize, lmbda, momentum=None):

        X_train, y_train = training_data
        X_val, y_val = validation_data
        X_test, y_test = test_data
        
        cost = self.cost(y) + lmbda * self.L2_sqr  
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

        idx = T.lscalar('idx') # miniBatch index
        
        # Compile function to train model.
        train_model = theano.function([idx, alpha], cost, updates=updates,
            givens = {
                X: X_train[idx*miniBatchSize: (idx+1)*miniBatchSize],
                y: y_train[idx*miniBatchSize: (idx+1)*miniBatchSize]
            }
        )

        # Compile functions to compute error on test set
        test_model = theano.function([idx], self.error(y),
           givens = {
               X: X_test[idx * miniBatchSize: (idx + 1) * miniBatchSize],
               y: y_test[idx * miniBatchSize: (idx + 1) * miniBatchSize]
           }
        )
        
        # Compile functions to compute error on test set
        validate_model = theano.function([idx], self.error(y),
           givens = {
               X: X_val[idx * miniBatchSize: (idx + 1) * miniBatchSize],
               y: y_val[idx * miniBatchSize: (idx + 1) * miniBatchSize]
           }
        )
        
        return train_model, validate_model, test_model


class convPoolLayer(object):

    def __init__(self,
                 rng,
                 filterShape,
                 imageShape,
                 poolShape=(2,2),
                 input=None,
                 input_dropout=None,
                 p_dropout=0.0,
                 activation=tanh):

        """ input/input_dropout is symbolic theano tensor of shape imageShape.

        filterShape is [number of filters, number of input feature maps, filter
        height, filter width],

        imageShape is batch size, number of input feature maps, image height, 
        image width.

        """
        assert filterShape[1] == imageShape[1]
        if p_dropout != 0.0:
            assert input_dropout is not None, 'To apply dropout input_dropout must be passed.'

        self.p_dropout = p_dropout
        self.activation = activation
        
        # Number of inputs to each unit in convolutional feature map 
        # (remembering that the input feature maps for a particular image
        # are summed over).
        fanIn = np.prod(filterShape[1:])

        # number of outputs from convolutional layer
        fanOut = filterShape[0] * np.prod(filterShape[2:]) / np.prod(poolShape)

        epsilon = np.sqrt(6./(fanIn+fanOut))  # from Xavier 2010 (tanh)
        
        W_values = np.asarray(
            rng.uniform(
                low = -epsilon,
                high = epsilon,
                size= filterShape),
            dtype=theano.config.floatX)

        # one value per convolutional feature map
        b_values = np.zeros(filterShape[0], dtype=theano.config.floatX)

        self.W = theano.shared(value = W_values, name = 'W', borrow = True)
        self.b = theano.shared(value = b_values, name = 'b', borrow = True)
        self.params = [self.W, self.b]

        # Set symbolic inputs and outputs.

        if input is not None: # layer is not a pure dropout layer
            self.input = input    
            conv_out = conv2d(input=self.input,
                              filters=(1-self.p_dropout)*self.W,
                              filter_shape=filterShape,
                              image_shape = imageShape)
        
            # batchSise by numFilters by height by width
            pool_out = downsample.max_pool_2d(input=conv_out, ds=poolShape,
                                              ignore_border=True)

            # b is reshaped ino shape [1, number of filters, 1, 1] so that 
            # each bias can be broadcasted across batch examples and units 
            # of corresponding feature map.
            self.output = self.activation(pool_out
                                 + self.b.dimshuffle('x', 0, 'x', 'x'))

        if input_dropout is not None:

            conv_out_dropout = conv2d(input=input_dropout,
                                      filters=self.W,
                                      filter_shape=filterShape,
                                      image_shape = imageShape)
            conv_out_dropout = dropout_from_layer(conv_out_dropout, self.p_dropout)
            pool_out_dropout = downsample.max_pool_2d(input=conv_out_dropout,
                                                      ds=poolShape,
                                                      ignore_border=True)
            self.output_dropout = self.activation(pool_out_dropout
                                         + self.b.dimshuffle('x', 0, 'x', 'x')) 

            

def apply_LeNet_sgd(training_data, validation_data, test_data, numFilters,
                    nLabels, dimHiddenSig, filterDim, poolShape,
                    miniBatchSize, sgd_opts, lmbda, dropout_rates, activations,
                    results_dir, momentum = None, mc_samples=None,
                    monitoring_to_file=False):
    
    """ X component of data sets passed in should be of shape [batch size, number 
    of input feature maps, image height, image width]
    """
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    os.chdir(results_dir)

    X = T.tensor4(name='X')
    y = T.ivector('y') 
    
    trainShape = training_data[0].get_value(borrow=True).shape
    valShape = validation_data[0].get_value(borrow=True).shape 
    testShape = test_data[0].get_value(borrow=True).shape

    nBatchTrain = trainShape[0]/miniBatchSize
    nBatchVal = valShape[0]/miniBatchSize
    nBatchTest = testShape[0]/miniBatchSize

    rng = np.random.RandomState(0)

    nChannels = trainShape[1]
    imRows = trainShape[2]
    imCols = trainShape[3]
    
     # Construct layers.
    
    layer0 = convPoolLayer(rng,
                           input = X,
                           #input_dropout = X,
                           imageShape = (miniBatchSize, nChannels,
                                         imRows, imCols),
                           filterShape = (numFilters[0], nChannels,
                                          filterDim, filterDim),
                           poolShape = poolShape,
                           p_dropout = dropout_rates[0],
                           activation=activations[0])

    
    imRows1 = (imRows-filterDim+1)/poolShape[0]
    imCols1 = (imCols-filterDim+1)/poolShape[1]
    
    layer1 = convPoolLayer(rng, 
                           input = layer0.output,
                           #input_dropout = layer0.output_dropout,
                           imageShape = (miniBatchSize, numFilters[0],
                                         imRows1, imCols1),
                           filterShape = (numFilters[1], numFilters[0],
                                          filterDim, filterDim),
                           poolShape = poolShape,
                           p_dropout = dropout_rates[1],
                           activation = activations[1])

    imRows2 = (imRows1-filterDim+1)/poolShape[0]
    imCols2 = (imCols1-filterDim+1)/poolShape[1]

    # One row of input is the flattened units of all feature maps
    # corresponding to a particular image.
    layer2 = FullyConnectedLayer(rng, 
                                 input = layer1.output.flatten(2),
                                 #input_dropout = layer1.output_dropout.flatten(2),
                                 dimIn = numFilters[1]*imRows2*imCols2,
                                 dimOut = dimHiddenSig,
                                 p_dropout = dropout_rates[2],
                                 activation = activations[2])

    layer3 = FullyConnectedLayer(rng, 
                                 input = layer2.output,
                                 #input_dropout = layer2.output_dropout,
                                 dimIn = layer2.dimOut,
                                 dimOut = dimHiddenSig,
                                 p_dropout = dropout_rates[3],
                                 activation = activations[3])

    layer4 = Softmax(rng,
                     input = layer3.output,
                     #input_dropout = layer3.output_dropout,
                     dimIn=layer3.dimOut,
                     dimOut=nLabels,
                     p_dropout = dropout_rates[4])

    # Initialise LeNet
    layers = [layer0, layer1, layer2, layer3, layer4]
    clf = cnn(rng, layers, mc_samples)
    
    train_model, validate_model, test_model = clf.get_training_functions(
        X,
        y,
        training_data,
        validation_data,
        test_data,
        miniBatchSize,
        lmbda,
        momentum)
        
    # Do the actual training.
    
    print('starting training...')
    
    [best_params, test_error, run_time, best_iter, temp_monitoring_filename] = supervised_tuning(
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

    
    # Save results

    trainingParams = {'sgd_opts':sgd_opts, 'lmbda':lmbda,
                      'numFilters':numFilters, 'dimHiddenSig': dimHiddenSig,
                      'miniBatchSize':miniBatchSize, 'filterDim':filterDim,
                      'poolShape':poolShape, 'dropout_rates':dropout_rates,
                      'momentum':momentum, 'mc_samples':mc_samples,
                      'activations':activations}

    results = ['test_error: {:.2%}'.format(test_error),
               'run_time: {:.2f}s'.format(run_time),
               'best_iter: {}'.format(best_iter)]

    with open('readme.txt', 'a') as f:
        f.write('\n {0}\n {1}\n'.format(trainingParams, results))
    f.close()

    if monitoring_to_file:
        monitoring_filename = 'LeNet_{0:.2%}_{1}_monitoring.txt'.format(
            test_error, numFilters)
        
        # rename temp file created during training
        os.rename(temp_monitoring_filename, monitoring_filename) 

        with open('readme.txt', 'a') as f:
            f.write('{0}\n'.format(monitoring_filename))
        f.close()

    while True:
        answer = raw_input('plot filters (y/n)?')
        if answer == 'y':
            # First nChannels rows relate to first filter etc.
            filters = best_params[0].reshape((-1, filterDim*filterDim))
            imageFilename = 'LeNet_filters_{0:.2%}_{1}.png'.format(
                test_error, numFilters)
            image = Image.fromarray(tile_raster_images(
                X=filters,
                img_shape=(filterDim, filterDim),
                tile_shape=(numFilters[0], nChannels),
                tile_spacing=(1, 1)))
            image.save(imageFilename)
            with open('readme.txt', 'a') as f:
                f.write('{0}\n'.format(imageFilename))
            f.close()
            break
        elif answer == 'n':
            break
        else:
            print('Invalid input, try again.')
        
    while True:    
        answer = raw_input('save model (y/n)? ')
        if answer == 'y':
            modelFilename = 'LeNet_{0:.2%}_{1}.pkl'.format(
                test_error, numFilters)
            best_model = {
                'imageShape': [nChannels, imRows, imCols], 
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

def test_saved_LeNet(X_test, nLabels,
                     filename,
                     label_descriptions,
                     plot_shape,
                     mc_dropout=False, y_test=None):

    f = open(filename, 'rb')
    model = cPickle.load(f)
    nChannels, imRows, imCols = model['imageShape']
    saved_params = model['params']
    numFilters = model['trainingParams']['numFilters']
    filterDim = model['trainingParams']['filterDim']
    poolShape = model['trainingParams']['poolShape']
    dimHiddenSig = model['trainingParams']['dimHiddenSig']
    activations = model['trainingParams']['activations']

    if mc_dropout:
        mc_samples = model['trainingParams']['mc_samples']
        dropout_rates = model['trainingParams']['dropout_rates']
    else:
        mc_samples = None

    X = T.tensor4('X')
    y = T.ivector('y')
    
    rng = np.random.RandomState(0)
    nSamples = X_test.get_value(borrow=True).shape[0]
    print (nChannels, imRows, imCols)
    print X_test.get_value(borrow=True).shape[1:]
    assert ((nChannels, imRows, imCols) == X_test.get_value(borrow=True).shape[1:]), 'dimension of test data samples incompatible with saved model'
    
    layer0 = convPoolLayer(rng,
                           input = X,
                           #input_dropout = X,
                           imageShape = (nSamples, nChannels,
                                         imRows, imCols),
                           filterShape = (numFilters[0], nChannels,
                                          filterDim, filterDim),
                           poolShape = poolShape,
                           p_dropout = dropout_rates[0],
                           activation = activations[0])

    imRows1 = (imRows-filterDim+1)/poolShape[0]
    imCols1 = (imCols-filterDim+1)/poolShape[1]
    
    layer1 = convPoolLayer(rng,
                           input=layer0.output,
                           #input_dropout = layer0.output_dropout,
                           imageShape = (nSamples, numFilters[0],
                                         imRows1, imCols1),
                           filterShape = (numFilters[1], numFilters[0],
                                          filterDim, filterDim),
                           poolShape = poolShape,
                           p_dropout = dropout_rates[1],
                           activation = activations[1])

    # Flatten units of all feature maps corresponding to a particular image
    # into a single array for input into (fully connected) sigmoid layer.

    imRows2 = (imRows1-filterDim+1)/poolShape[0]
    imCols2 = (imCols1-filterDim+1)/poolShape[1]

    layer2 = FullyConnectedLayer(rng,
                                 input=layer1.output.flatten(2),
                                 #input_dropout = layer1.output_dropout.flatten(2),
                                 dimIn = numFilters[1]*imRows2*imCols2,
                                 dimOut = dimHiddenSig,
                                 p_dropout = dropout_rates[2],
                                 activation=activations[2])



    layer3 = FullyConnectedLayer(rng, 
                                 input = layer2.output,
                                 #input_dropout = layer2.output_dropout,
                                 dimIn = layer2.dimOut,
                                 dimOut = dimHiddenSig,
                                 p_dropout = dropout_rates[3],
                                 activation = activations[3])

    
    layer4 = Softmax(rng,
                     input = layer3.output,
                     #input_dropout = layer3.output_dropout,
                     dimIn=layer2.dimOut,
                     dimOut=nLabels,
                     p_dropout = dropout_rates[4])

    # Initialise LeNet

    layers = [layer0, layer1, layer2, layer3, layer4]
    clf = cnn(rng, layers, mc_samples)
    
    # Set params
    for param, saved_param in zip(clf.params, saved_params):
        param.set_value(saved_param, borrow=True)


    if mc_dropout:
        get_predictions = theano.function([], [clf.predictive_distribution_mean,
                                               clf.predictive_distribution_var],
                                          givens = {
                                              X:X_test,
                                          }
        )

        predictive_distribution_mean, predictive_distribution_var = get_predictions()

        # misclassified = np.nonzero(
        #     np.not_equal(
        #         np.argmax(
        #             predictive_distribution_mean, axis=1),
        #         y_test.eval()
        #     )
        # )
        
        plot_predictions(predictive_distribution_mean[:, [2,3]],
                         predictive_distribution_var[:, [2,3]],
                         colors = ['g','r'], labels=['grass', 'other'],
                         xtick_labels=['0','10','20','30','40','50','60','70','80','90','100'],
                         x_label='Corruption (%)')

    else:
        get_predictions = theano.function([], clf.preds,
                                          givens = {
                                              X:X_test,
                                          }
        )

        print 'Generating Predictions...'
        predictions = get_predictions()

        if y_test is not None:

            assert nSamples == y_test.eval().shape[0]
            assert y_test.eval().max() <= nLabels-1 and y_test.eval().min() >= 0

            test_model = theano.function([], clf.error(y),
                givens = {
                    X:X_test,
                    y:y_test
                }
            )

            print 'Computing model error on test set...'
            test_error = float(test_model())
        
            
            print('test score {0:.2%}'
                  .format(test_error)
            )

            assert plot_shape[0]*plot_shape[1] == nSamples
            print 'Plotting ground-truth and predictions'
            plot_labels_as_image(
            [y_test.eval().reshape(plot_shape), predictions.reshape(plot_shape)],
            titles=['Ground-Truth', 'Predictions'],
            cmap='terrain',
            nColors=nLabels)

            print 'Plotting confusion matrix...'
            confusion = confusion_matrix(y_test.eval(), predictions)
            plot_confusion_matrix(confusion, label_descriptions)
            
        else:

            print 'Plotting predictions...'
            plot_labels_as_image(predictions.reshape(plot_shape),
                                 cmap='terrain',
                                 nColors=nLabels,
                                 zoomParams=zoomParams)

    
        
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

        numFilters = [6, 12]
        nLabels = 4
        dimHiddenSig = 128
        filterDim = 5
        poolShape = (3,3)
        sgd_opts = {'min_epochs':50, 'max_epochs':50, 'alpha_init':0.01,
                    'gamma':0.0001, 'p':0.75, 'monitor_frequency':1000}
        miniBatchSize = 20
        lmbda = 0.0
        dropout_rates = [0.0, 0.0, 0.0, 0.0, 0.0]
        activations = [relu, relu, relu, relu]
        momentum = 0.9
        mc_samples = None
        monitoring_to_file = True

        try:
            print 'loading in data...'
            train_set, validation_set, test_set = load_data_npz(data_file)

        except IOError:
            print 'cannot open data_file', data_file


        train_set, validation_set, test_set = [toShared_xy(train_set),
                                               toShared_xy(validation_set),
                                               toShared_xy(test_set)]
        print 'setting up model...'
        apply_LeNet_sgd(
            train_set, validation_set, test_set, numFilters, nLabels,
            dimHiddenSig, filterDim, poolShape, miniBatchSize, sgd_opts,
            lmbda, dropout_rates, activations, results_dir, momentum,
            mc_samples, monitoring_to_file)
    
    elif model_file is not None:

        # Test a saved model.

        nLabels = 4
        plot_shape = (100,100)
        nSamples = 10000
        label_descriptions=['barren', 'trees', 'grassland', 'other']
        
        try:
            test_set = load_data_npz(data_file, only_test=True)
        except IOError:
            print 'cannot open data_file', data_file

        X_test, y_test = toShared_x(test_set)
        print 'testing saved model...'
        test_saved_LeNet(X_test, nLabels, model_file,
                         label_descriptions, plot_shape,
                         mc_dropout=True)
        
    else:
        print 'exactly 2 arguments must be passed on command line'
        print 'usage1: softmax.py -d <data_file> -r <results_dir>'
        print 'usage2: softmax.py -d <data_file> -m <saved_model_file>'


if __name__ == "__main__":

    main(sys.argv[1:])



    
        


        
        
    
