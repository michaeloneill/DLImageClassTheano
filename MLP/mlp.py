import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from theano.sandbox import cuda
from PIL import Image

from DeepLearning.softmax.softmax import Softmax
from DeepLearning.common.loadData import load_data_npz, toShared_xy, toShared_x
from DeepLearning.common.plotting import tile_raster_images, plot_labels_as_image, plot_bar, plot_confusion_matrix, plot_neuron_space_class_signatures
from DeepLearning.common.training import supervised_tuning, dropout_from_layer, relu

from sklearn.metrics import confusion_matrix

import cPickle
import os
import sys, getopt
import timeit


class MLP(object):

    def __init__(self, rng, dimIn, dimHiddenLayers, dimOut,
                 dropout_rates, activations):

        self.X = T.matrix('X')
        self.y = T.ivector('y')
        self.dimIn = dimIn
        self.dimOut = dimOut

        self.nHiddenLayers = len(dimHiddenLayers)
        self.layers = []
        self.params = []

        for i in xrange(self.nHiddenLayers):
            if i==0:
                layerDimIn = dimIn
                layerInput = self.X
                if dropout_rates[0]!=0.0:
                    layerInputDropout = self.X
                else:
                    layerInputDropout = None
            else:
                layerDimIn = dimHiddenLayers[i-1]
                layerInput = self.layers[-1].output
                try:
                    layerInputDropout = self.layers[-1].output_dropout
                except:
                    if dropout_rates[i]!=0.0:
                        layerInputDropout=self.layers[-1].output
                    else:
                        layerInputDropout = None

            layer = FullyConnectedLayer(rng,
                                        dimIn=layerDimIn,
                                        dimOut=dimHiddenLayers[i],
                                        input=layerInput,
                                        input_dropout=layerInputDropout,
                                        p_dropout=dropout_rates[i],
                                        activation=activations[i])
            
            self.layers.append(layer)
            self.params.extend(layer.params)

        # set up softmax layer

        softmaxDimIn = dimHiddenLayers[-1]
        softmaxInput = self.layers[-1].output

        try:
            softmaxInputDropout = self.layers[-1].output_dropout
        except:
            if dropout_rates[-1] != 0.0:
                softmaxInputDropout = self.layers[-1].output
            else:
                softmaxInputDropout = None
            
        softmax_layer = Softmax(rng,
                                dimIn=softmaxDimIn,
                                dimOut=dimOut,
                                input=softmaxInput,
                                input_dropout=softmaxInputDropout,
                                p_dropout=dropout_rates[-1])

        self.layers.append(softmax_layer)
        self.params.extend(softmax_layer.params)

        self.cost = self.layers[-1].cost  # fn pointer
        self.preds = self.layers[-1].preds
        self.error = self.layers[-1].error  # fn pointer

        self.L1 = (
            T.sum([abs(layer.W).sum() for layer in self.layers]) 
        )
        self.L2_sqr = (
            T.sum([(layer.W ** 2).sum() for layer in self.layers])
        )


    def get_training_functions(self, training_data, validation_data,
                               test_data, miniBatchSize, lmbda, momentum=None):

        X_train, y_train = training_data
        X_val, y_val = validation_data
        X_test, y_test = test_data

        cost = self.cost(self.y) + lmbda*self.L2_sqr
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
    
        # Compile function to train model.
        train_model = theano.function([idx, alpha], cost, updates=updates,
            givens = {
                self.X: X_train[idx*miniBatchSize: (idx+1)*miniBatchSize],
                self.y: y_train[idx*miniBatchSize: (idx+1)*miniBatchSize],
            }
        )

        # Compile function to compute error on test set.
        test_model = theano.function([idx], self.error(self.y),
            givens = {
                self.X: X_test[idx * miniBatchSize: (idx + 1) * miniBatchSize],
                self.y: y_test[idx * miniBatchSize: (idx + 1) * miniBatchSize]
            }
        )

        # Compile function to compute error on validation set.
        validate_model = theano.function([idx], self.error(self.y),
            givens = {
                self.X: X_val[idx * miniBatchSize: (idx + 1) * miniBatchSize],
                self.y: y_val[idx * miniBatchSize: (idx + 1) * miniBatchSize]
            }
        )
                                        
        return train_model, validate_model, test_model
    

class FullyConnectedLayer(object):

    def __init__(self, rng, dimIn, dimOut,
                 input=None, input_dropout=None, p_dropout=0.0, activation=sigmoid):

        if p_dropout != 0.0:
            assert input_dropout is not None, 'No input dropout passed'
        
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.p_dropout = p_dropout
        self.activation = activation

        epsilon = np.sqrt(6./(dimIn+dimOut))  # from Xavier 2010 (tanh or ReLU)
        
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
        
        if input is not None:  # layer is not a pure dropout layer
            self.input = input
            self.output = self.activation( 
                (1-self.p_dropout)*T.dot(self.input, self.W) + self.b)
            
        if input_dropout is not None:
            self.input_dropout = dropout_from_layer(
                input_dropout, p_dropout) 
            self.output_dropout = self.activation(
                T.dot(self.input_dropout, self.W) + self.b)

            
def apply_mlp_sgd(training_data, validation_data, test_data,
                  dimHiddenLayers, nLabels, miniBatchSize,
                  sgd_opts, lmbda, dropout_rates, activations,
                  results_dir, momentum=None, monitoring_to_file=False):
                                        

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    os.chdir(results_dir)

    trainShape = training_data[0].get_value(borrow=True).shape
    valShape = validation_data[0].get_value(borrow=True).shape 
    testShape = test_data[0].get_value(borrow=True).shape

    nBatchTrain = trainShape[0]/miniBatchSize
    nBatchVal = valShape[0]/miniBatchSize
    nBatchTest = testShape[0]/miniBatchSize

    rng = np.random.RandomState(0)

    clf = MLP(rng,
              trainShape[1],
              dimHiddenLayers,
              nLabels,
              dropout_rates,
              activations)

    train_model, validate_model, test_model = clf.get_training_functions(
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
                      'miniBatchSize':miniBatchSize,
                      'dimIn':trainShape[1],
                      'dimHiddenLayers':dimHiddenLayers,
                      'dimOut':nLabels,
                      'dropout_rates':dropout_rates,
                      'momentum':momentum,
                      'activations':activations}

    results = ['test_error: {:.2%}'.format(test_error),
               'run_time: {:.2f}s'.format(run_time),
               'best_iter: {}'.format(best_iter)]

    with open('readme.txt', 'a') as f:
        f.write('\n {0}\n {1}\n'.format(trainingParams, results))
    f.close()

    if monitoring_to_file:
        monitoring_filename = 'MLP_{0:.2%}_{1}_monitoring.txt'.format(
            test_error, dimHiddenLayers )
        
        # rename temp file created during training
        os.rename(temp_monitoring_filename, monitoring_filename) 

        with open('readme.txt', 'a') as f:
            f.write('{0}\n'.format(monitoring_filename))
        f.close()

    while True:
        answer = raw_input('plot filters (y/n)?')
        if answer == 'y':
            filters = best_params[0].T
            imageFilename = 'MLP_filters_{0:.2%}_{1}.png'.format(
                test_error, dimHiddenLayers)
            dimImage = int(np.sqrt(trainShape[1]))
            image = Image.fromarray(tile_raster_images(
                X=filters,
                img_shape=(dimImage, dimImage),
                tile_shape=(100, 100), # plot first 100
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
            modelFilename = 'MLP_{0:.2%}_{1}.pkl'.format(
                test_error, dimHiddenLayers)
            best_model = {
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



def test_saved_model(X_test, filename, label_descriptions, imageShape, miniBatchSize, y_test=None, zoomParams=None):

    """ we test in batches in case the test set is too large to fit in memory """
    
    f = open(filename, 'rb')
    model = cPickle.load(f)
    saved_params = model['params']
    dimIn = model['trainingParams']['dimIn']
    dimHiddenLayers = model['trainingParams']['dimHiddenLayers']
    dimOut = model['trainingParams']['dimOut']
    activations = model['trainingParams']['activations']

    np_rng = np.random.RandomState(0)
    
    testShape = X_test.get_value(borrow=True).shape
    nBatchTest = testShape[0]/miniBatchSize
    remainder = testShape[0]-miniBatchSize*nBatchTest
    
    assert dimIn == testShape[1]

    dropout_rates = [0.0 for i in xrange(len(dimHiddenLayers)+1)]

    clf = MLP(np_rng,
              dimIn,
              dimHiddenLayers,
              dimOut,
              dropout_rates,
              activations)

    for param, saved_param in zip(clf.params, saved_params):
        param.set_value(saved_param, borrow=True)


    # Plot filters for first layer

    print 'plotting filters for first layer...'
    
    filters = saved_params[0].T
    # temp = np.floor(np.sqrt(dimIn))
    # img_shape = (int(temp), int(np.ceil(dimIn/temp)))
    img_shape = (1,dimIn)
    image = Image.fromarray(tile_raster_images(
        X=(filters,np.ones(shape=filters.shape), None, None),
        img_shape=img_shape,
        tile_shape=(10,10), #plot first 100
        tile_spacing=(1,1)))
    image.show()


    idx = T.lscalar('idx')

    # generate predictions
    
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


    if y_test is not None:
        assert X_test.get_value(borrow=True).shape[0]==y_test.eval().shape[0]
        assert y_test.eval().max() <= dimOut-1 and y_test.eval().min()>=0


        test_model = theano.function([idx], clf.error(clf.y),
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

        print 'Plotting confusion matrix...'
        confusion = confusion_matrix(y_test.eval(), predictions)
        plot_confusion_matrix(confusion, label_descriptions)


        print 'Computing and plotting first layer neuron class averages...'
        get_first_layer_outputs = theano.function([idx], clf.layers[0].output,
            givens = {
                clf.X:X_test[idx*miniBatchSize:(idx+1)*miniBatchSize]
            }
        )
        get_first_layer_outputs_remainder = theano.function([], clf.layers[0].output,
            givens = {
                clf.X:X_test[nBatchTest*miniBatchSize:testShape[0]]
            }
        )
        first_layer_outputs = np.concatenate(tuple(get_first_layer_outputs(i)
                                               for i in xrange(nBatchTest))
                                         +(get_first_layer_outputs_remainder(),))
        print first_layer_outputs.shape
        first_layer_class_averages = np.array(
            [np.mean(first_layer_outputs[y_test.eval()==i],axis=0)
             for i in range(dimOut)])
        print first_layer_class_averages.shape

        plot_neuron_space_class_signatures(first_layer_class_averages, label_descriptions)
        
        
        print 'Plotting ground-truth and predictions'
        plot_labels_as_image(
            [y_test.eval().reshape(imageShape), predictions.reshape(imageShape)],
            titles=['Ground-Truth', 'Predictions'],
            cmap='terrain',
            nColors=dimOut,
            label_descriptions=label_descriptions,
            zoomParams=zoomParams)

        
    else:

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
        opts, args = getopt.getopt(
            argv, "hd:r:m:", ["data_file=", "results_dir="])
    except getopt.GetoptError:
        print 'incorrect usage'
        print 'usage: mlp.py -d <data_file> -r <results_dir>'
        sys.exit(2)
        
    for opt, arg in opts:
        if opt=="-h":
            print 'usage1: mlp.py -d <data_file> -r <results_dir>'
        elif opt in ("-d", "--data_file"):
            data_file = arg
        elif opt in ("-r", "--results__dir"):
            results_dir = arg
        elif opt in ("-m", "--model_file"):
            model_file = arg
            
    if data_file is None:
        print 'data_file was not given'
        print 'usage1: mlp.py -d <data_file> -r <results_dir>'
        print 'usage2: mlp.py -d <data_file> -m <saved_model_file>'
        sys.exit(2)

    if results_dir is not None:

        # train a new model

        dimHiddenLayers = [100, 100, 100]
        nLabels = 4
        miniBatchSize = 20
        sgd_opts = {'min_epochs':10, 'max_epochs':10, 'alpha_init':0.1,
                    'gamma':0.0001, 'p':0.75, 'monitor_frequency':1000}
        lmbda = 0.0
        dropout_rates = [0.0, 0.0, 0.0, 0.5]
        activations = [relu, relu, relu]
        momentum = 0.9
        monitoring_to_file = True
        
        label_descriptions = ['rapeseed', 'water', 'built up', 'bare soil', 'wheat', 'grass', 'clouds', 'cloud shadows']
        
        try:
            print 'loading in data...'
            data = load_data_npz(data_file)
            #plot_bar(np.bincount(train_set[1]), xlabel='landcover class', ylabel='number of samples', label_descriptions=label_descriptions)
        
            # landsat 2 remove class that corresponds to border
            # for i in xrange(len(data)):
            #     keep = data[i][1]!=13
            #     data[i][0]=data[i][0][keep]
            #     data[i][1]=data[i][1][keep]
            
            #     data[i][1][data[i][1]==14]=13
            #     data[i][1][data[i][1]==15]=14
            #     data[i][1][data[i][1]==16]=15
            #     print np.unique(data[i][1])

            train_set, validation_set, test_set = data
            
        except IOError:
            print 'cannot open data_file', data_file

        
        train_set, validation_set, test_set = [toShared_xy(train_set),
                                               toShared_xy(validation_set),
                                               toShared_xy(test_set)]
        print 'setting up model...'
        apply_mlp_sgd(
            train_set,
            validation_set,
            test_set,
            dimHiddenLayers,
            nLabels,
            miniBatchSize,
            sgd_opts,
            lmbda,
            dropout_rates,
            activations,
            results_dir,
            momentum,
            monitoring_to_file)


    elif model_file is not None:

        # Test a saved model.

        # rapideye
        imageShape = (5000,5000) 
        zoomParams = {'zoom':10, 'x1':2400, 'x2':2600,
                      'y1':2400, 'y2':2600} # origin top left
        label_descriptions = ['rapeseed', 'water', 'built up', 'other', 'wheat', 'grass', 'clouds', 'cloud shadows']

        #landsat2
        # imageShape = (8191,8081)
        # zoomParams = {'zoom':30, 'x1':5100, 'x2':5200,
        #               'y1':3100, 'y2':3200} # origin top left
        # label_descriptions = ['rapeseed', 'wheat', 'grass 1', 'grass 2', 'grass 3', 'built up 1', 'built up 2', 'fallow', 'bare soil 1', 'barley', 'built up 3', 'potatoes', 'shallow water', 'bare soil 2', 'clouds', 'cloud shadows']

       
        miniBatchSize = 1000

        try:
            test_set = load_data_npz(data_file, only_test=True)

            # landsat 2 remove class that corresponds to border
            # keep = test_set[1]!=13
            # test_set[0]=test_set[0][keep]
            # test_set[1]=test_set[1][keep]
            
            # test_set[1][test_set[1]==14]=13
            # test_set[1][test_set[1]==15]=14
            # test_set[1][test_set[1]==16]=15
            # print np.unique(test_set[1])


        except IOError:
            print 'cannot open data_file', data_file

        X_test, y_test = toShared_xy(test_set)
        print 'testing saved model...'
        test_saved_model(X_test, model_file,
                         label_descriptions, imageShape,
                         miniBatchSize, y_test, zoomParams)
        
    else:
        print 'exactly 2 arguments must be passed on command line'
        print 'usage1: mlp.py -d <data_file> -r <results_dir>'
        print 'usage2: mlp.py -d <data_file> -m <saved_model_file>'


if __name__ == "__main__":

    main(sys.argv[1:])


