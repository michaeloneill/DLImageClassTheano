import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from PIL import Image

import cPickle
import os
import sys, getopt

import timeit
from DeepLearning.common.loadData import load_data_pickled, load_data_npz, toShared_x
from DeepLearning.common.plotting import tile_raster_images


class dA(object):

    # denoising auto-encoder class
    
    def __init__(self, np_rng, input, dimIn, dimHidden, theano_rng=None, W=None, bEncode=None, bDecode=None):

        """ np_rng used for weights initialisations. theano_rng used to corrupt input 
        data."""

        """ In the case that multiple dAs are stacked into a sdA for unsupervised 
        pre-training, W, bEncode and bDecode are the parameters that are shared with 
        the fine-tuning supervised architecture i.e. the optimal sdA parameters learned 
        in the pre-training phase will initialise those of the architecture in the 
        fine-tuning stage. In the case that the dA is used stand-alone, these will be 
        set to None."""

        if not theano_rng:
            theano_rng = RandomStreams(123)

        if not W:
            epsilon = 4*np.sqrt(6./(dimIn+dimHidden)) # from Xavier 2010 (sigmoid)
            W_values = np.asarray(
                np_rng.uniform(
                    low = -epsilon,
                    high = epsilon,
                    size=(dimIn, dimHidden)),
                dtype=theano.config.floatX)
            
            W = theano.shared(value = W_values, name = 'W', borrow = True)
        
        if not bEncode: 
            bEncode_values = np.zeros(dimHidden, dtype=theano.config.floatX)
            bEncode = theano.shared(
                value = bEncode_values, name = 'bEncode', borrow = True)
                
        if not bDecode:
            bDecode_values = np.zeros(dimIn, dtype=theano.config.floatX)
            bDecode = theano.shared(
                value = bDecode_values, name = 'bDecode', borrow = True)
                
        self.W = W
        self.bEncode = bEncode
        self.bDecode = bDecode
        self.params = [self.W, self.bEncode, self.bDecode]
        self.theano_rng = theano_rng

        # symbolic inputs and outputs
        
        self.input = input
        
    def corruptInput(self, input, corruptLevel):
        """ A ramdomly selected subset of corruptLevel of the input values are zeroed 
        out. The rest are left unchanged. """

        return self.theano_rng.binomial(
            size=input.shape,
            n=1,
            p=1-corruptLevel, # prob of a 1 (corresponding value unchanged)
            dtype=theano.config.floatX)*input
    
    def encode(self, input):
        """ encodes input """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.bEncode)
    
    def decode(self, y):
        # y is the encoded representation
        return T.nnet.sigmoid(T.dot(y, self.W.T) + self.bDecode) 


        
    def cost(self, corruptLevel):
        """ returns a measure of how well reconstrcted corrupted input matches 
        the raw input """
        
        input_corrupted = self.corruptInput(self.input, corruptLevel)
        y = self.encode(input_corrupted)
        z = self.decode(y)

        #x entropy cost (for binary units)
        return -T.mean(T.sum(self.input * T.log(z) + (1 - self.input) * T.log(1 - z),
                            axis=1))
        # sse cost
        #return T.mean((self.input-z)**2)

    def get_cost_updates(self, corruptLevel, alpha, momentum=None):

        cost = self.cost(corruptLevel)
        grads = T.grad(cost, self.params)

        if momentum is None:
            updates = [
                (param, param - alpha*grad)
                for param, grad in zip(self.params, grads)
            ]

        else:
            assert (momentum >= 0 and momentum < 1)
            updates = []
            for param, grad in zip(self.params, grads):
                # Create shared variable to store parameter update.
                # Note this is only initialised once (on first iteration of SGD)
                momentum_update = theano.shared(param.get_value()*0.,
                                             broadcastable=param.broadcastable)
                # param_update refers to the updated param_update.
                updates.append((param, param - alpha*momentum_update))
                # Mix in previous (negative) step direction with current gradient.
                updates.append((momentum_update, momentum*momentum_update
                                + (1.-momentum)*grad)) 

        return cost, updates

        
def apply_dA_sgd(X_train, nChannels, dimHidden, epochs, miniBatchSize, alpha, results_dir, momentum=None):


    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    os.chdir(results_dir)
    
    X = T.matrix('X')
    np_rng = np.random.RandomState(0)
    theano_rng = RandomStreams(123)
    corruptLevel = 0.3
    nBatches = X_train.get_value().shape[0]/miniBatchSize

    dimIn = X_train.get_value(borrow=True).shape[1]
    dALayer = dA(np_rng = np_rng, input = X, dimIn = dimIn, dimHidden = dimHidden)

    
    cost, updates = dALayer.get_cost_updates(corruptLevel, alpha, momentum)
    
    idx = T.lscalar('idx') # minibatch index
    
    train_model = theano.function([idx], cost, updates=updates,
        givens = {
            X: X_train[idx*miniBatchSize: (idx+1)*miniBatchSize]
        }
    )

    # do the actual training

    print('starting training of dA with corruptiion level {0}...'.format(corruptLevel))
    start_time = timeit.default_timer()

    for epoch in xrange(epochs):
        totalCost = 0.0
        for miniBatchIndex in xrange(nBatches):
            totalCost += train_model(miniBatchIndex) # aggregate miniBatch costs
        totalCost /= nBatches # average over miniBatches
        print('cost after epoch {0}: {1:.2f}'.format(epoch+1, totalCost))

    end_time = timeit.default_timer()
    run_time = '{0:.2f}'.format(end_time-start_time)

    opt_params = [param.get_value(borrow=False) for param in dALayer.params]


    # save results


    if nChannels>1:
        dimImage = int(np.sqrt(dimIn/nChannels)) # assume square image
        tile_shape = (10, nChannels) # plot first 10 filters across each channel
        X=opt_params[0].T.reshape(-1, dimImage*dimImage) # first nChannels rows relate to first filter
    else:
        dimImage = int(np.sqrt(dimIn)) # assume square image
        tile_shape = (10, 10) # or just plot first 100 filters
        X=opt_params[0].T
    
    imageFilename = 'dA_filters_{0}_corruption_{1}.png'.format(
        dimHidden, corruptLevel)

    image = Image.fromarray(tile_raster_images(
        X=X,
        img_shape=(dimImage, dimImage), tile_shape=tile_shape,
        tile_spacing=(1, 1)))
    image.save(imageFilename)
    
    
    
    trainingParams = {'epochs':epochs, 'miniBatchSize':miniBatchSize,
                      'alpha':alpha, 'dimHidden':dimHidden,
                      'corruptLevel':corruptLevel}
    
            
    with open('readme.txt', 'a') as f:
        f.write('\n {0}\n runtime: {1}\n {2}\n'.format(
            trainingParams, run_time,
            imageFilename))
    f.close()
                
    os.chdir('../')
    

def main(argv):

    data_file = None
    results_dir = None

    try:
        opts, args = getopt.getopt(argv, "hd:r:", ["data_file=", "results_dir="])
    except getopt.GetoptError:
        print 'incorrect usage'
        print 'usage: softmax.py -d <data_file> -r <results_dir>'
        sys.exit(2)
        
    for opt, arg in opts:
        if opt=="-h":
            print 'usage: softmax.py -d <data_file> -r <results_dir>'

        elif opt in ("-d", "--data_file"):
            data_file = arg
        elif opt in ("-r", "--results__dir"):
            results_dir = arg
            
    if data_file is None:
        print 'data_file was not given'
        print 'usage: softmax.py -d <data_file> -r <results_dir>'
        sys.exit(2)

    try:
        train_set, validation_set, test_set = load_data_npz(data_file) 
    except IOError:
        print 'cannot open data_file', data_file

    X = train_set[0]
    X = toShared_x(X)

    nChannels = 4
    dimHidden = 100
    epochs = 15
    miniBatchSize = 20
    alpha = 0.001
    momentum=None

    apply_dA_sgd(X, nChannels, dimHidden, epochs, miniBatchSize, alpha, results_dir, momentum)



if __name__ == "__main__":

    main(sys.argv[1:])


    
    
    
    

