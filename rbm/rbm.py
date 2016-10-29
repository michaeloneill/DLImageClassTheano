import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

import timeit
from PIL import Image
import os
import sys, getopt
import cPickle


from DeepLearning.common.loadData import load_data_pickled, load_data_npz, toShared_x
from DeepLearning.common.plotting import tile_raster_images

class rbm(object):

    def __init__(self, np_rng, input, dimIn, dimHidden, theano_rng=None, W=None, b_v=None, b_h=None):

        self.dimIn = dimIn
        self.dimHidden = dimHidden
        
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
        
        if not b_h: 
            b_h_values = np.zeros(dimHidden, dtype=theano.config.floatX)
            b_h = theano.shared(
                value = b_h_values, name = 'b_h', borrow = True)
                
        if not b_v:
            b_v_values = np.zeros(dimIn, dtype=theano.config.floatX)
            b_v = theano.shared(
                value = b_v_values, name = 'b_v', borrow = True)
                
        self.W = W
        self.b_h = b_h
        self.b_v = b_v
        self.params = [self.W, self.b_h, self.b_v]
        self.theano_rng = theano_rng

        self.input = input

    def h_given_v(self, v):
        """ returns hidden expectation given visible input """
        """ Also returns the pre-sigmoid activation for theano graph optimisation """
        z = T.dot(v, self.W) + self.b_h
        return [z, T.nnet.sigmoid(z)]
    
    def v_given_h(self, h):
        """ returns visible expectation given hidden units """
        z = T.dot(h, self.W.T) + self.b_v 
        return [z, T.nnet.sigmoid(z)] 

    def sample_h_given_v(self, v0_sample):
        """ symbolic binomial sampling (outcome 1 or 0 as this is a binary RBM) 
        taking hidden expectation as a probability """

        h1_mean_pre_sig, h1_mean = self.h_given_v(v0_sample)

        # the following returns a symbolic variable
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1,
                                             p=h1_mean, dtype=theano.config.floatX)
        return [h1_mean_pre_sig, h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        """ symbolic binomial sampling (outcome 1 or 0 as this is a binary RBM) 
        taking visible expectation as a probability """

        v1_mean_pre_sig, v1_mean = self.v_given_h(h0_sample)

        # the following returns a symbolic variable
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1,
                                             p=v1_mean, dtype=theano.config.floatX)
        return [v1_mean_pre_sig, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """ gibbs sampling step starting from hidden. Useful for performing 
        CD and PCD updates """

        v1_pre_sig, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_pre_sig, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_pre_sig, v1_mean, v1_sample,
                h1_pre_sig, h1_mean, h1_sample]
    
    def gibbs_vhv(self, v0_sample):
        """ gibbs sampling step starting from visible. Useful for performing 
        sampling from RBM """

        h1_pre_sig, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_pre_sig, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)

        return [h1_pre_sig, h1_mean, h1_sample,
                v1_pre_sig, v1_mean, v1_sample]

    def free_energy(self, v_sample):
        """ Remember this is only for binary RBMs"""
        
        hidden_term_exponent = T.dot(v_sample, self.W) + self.b_h
        hidden_term = T.sum(T.log(1+T.exp(hidden_term_exponent)), axis=1)
        visible_term = T.dot(v_sample, self.b_v)
        
        return -hidden_term - visible_term

    def get_cost_updates(self, alpha, persistent, k):
        """ Implements one step of CD-k or PCD-k
        When persistent=None, CD-k sampling executed using a newly
        generated hidden sample. Otherwise, PCD-k is executed
        taking persistent to be a shared variable containing
        old state of Gibbs chain (size=[batch size, number 
        of hidden units]). K is number of Gibbs steps
        to perform. Returns a proxy for the cost (the true
        cost is too expensive to compute) and the updates:
        containing updates for weights, biases, and 
        persistent shared variable. """

        # compute positive phase sample (see form of positive phase gradient for binary rbm) 
        h_pre_sig_pos, h_mean_pos, h_sample_pos = self.sample_h_given_v(self.input)

        if persistent is None:
        # Perform CD-k: the newly generated hidden sample initialises
        # the gibbs chain. This sample was generated from an actual training
        # sample from the input (i.e. a sample from a distribution expected to be
        # close to the generated distribution) as per CD.

            chain_start = h_sample_pos
        else:
            # perform PCD instead, initialsing the chain from the persistent state
            # after preceding gibbs step 
            chain_start = persistent

        # Set up performance of sampling for negative phase
        # with symbolic scan performing k iterations of gibbs.
        # chain_start is the initial state corresponding to the 6th output,
        # updates. This contains, in a dictionary, the update rules
        # to the shared variables used by the function in scan (in this case the sample),
        # so that successive applications of scan use the updated shared variable (sample)
        # from the last application.
        # The other initialisations are place-holders - they don't require
        # initialising

        ([v_pre_sig_neg, v_mean_neg, v_sample_neg,
          h_pre_sig_neg, h_mean_neg, h_sample_neg],
         updates) = theano.scan(self.gibbs_hvh,
                                outputs_info=[None, None, None, None, None,
                                                   chain_start],
                                n_steps=k,
                                name="gibbs_hvh")
                                
        # Now we take the sample at the end of the chain to get
        # the free energy of the negative phase

        chain_end = v_sample_neg[-1]

        # proxy for the neg log likelihood
        # cost.  
        cost = T.mean(self.free_energy(self.input))-T.mean(
            self.free_energy(chain_end))
        

        # Note that chain_end is a symbolic variable expressed
        # in terms of the model parameters. So naive application
        # of T.grad will try to go through Gibbs chain. We specify it
        # as a constant to prevent this #

        grads = T.grad(cost, self.params, consider_constant=[chain_end])

        # add parameter updates to updates returned by scan

        for grad, param in zip(grads, self.params):
            updates[param] = param-grad*T.cast(
                alpha, dtype=theano.config.floatX
            ) # cast alpha to correct type if not already

        # construct costs for monitoring, sice we can't evaluate the
        # neg log likelihood cost due to intractable z (and the above
        # proxy is not a sesible  measure of the latter either).
            
        if persistent:
            # Add persistent state to updates
            # persistent must be shared!
            updates[persistent] = h_sample_neg[-1]
            # for PCD_k we use pseudo-likelihood
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)

        else:
            # for CD-k we use reconstruction cross-entropy
            monitoring_cost = self.get_reconstruction_cost(
                v_pre_sig_neg[-1])
            
        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        
        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)
        
        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)
        
        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        
        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)
        
        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost_pseudo_likelihood = T.mean(self.dimIn * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))
        
        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.dimIn
        
        return cost_pseudo_likelihood


    def get_reconstruction_cost(self, vs_pre_sig_neg):
        """The reconstruction error for monitoring during training.
        Note that this is not the function being approximated through CD-k.
        The function requires the pre-sigmoid activation as
        input, so that Theano sees log(sigmoid) rather than log(scan)  """

        
        cost_X_entropy = -T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(vs_pre_sig_neg)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(vs_pre_sig_neg)),
                axis=1
            )
        )
        
        return cost_X_entropy
                            
    
def apply_rbm_sgd(X_train, X_test, nChannels, dimHidden, epochs, miniBatchSize, alpha, nChains,
                  n_plotting_samples, persistent_bool, k, results_dir):

    """ nChains is the number of chains to run simultaneously (number of samples to pass
    to gibbs. nPlotSamples is number of samples to plot for each chain) """
    
    # create/navigate to output folder
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    os.chdir(results_dir)


    X = T.matrix('X')
    dimIn = X_train.get_value(borrow=True).shape[1]

    # whether to use PCD-k or CD-k
    if persistent_bool:
        persistent_chain = theano.shared(
            np.zeros(
                (miniBatchSize, dimHidden),
                dtype=theano.config.floatX),
            borrow=True)
    else:
        persistent_chain = None
        
    np_rng = np.random.RandomState(0)
    theano_rng = RandomStreams(123)
    nBatches = X_train.get_value().shape[0]/miniBatchSize

    RBM = rbm(np_rng, input=X, dimIn=dimIn,
                dimHidden=dimHidden, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-k
    cost, updates = RBM.get_cost_updates(alpha=alpha,
                                         persistent=persistent_chain, k=k)
    
    idx = T.lscalar('idx') # minibatch index
    
    train_model = theano.function([idx], cost, updates=updates,
        givens = {
            X: X_train[idx*miniBatchSize: (idx+1)*miniBatchSize]
        }
    )
    
    # do the actual training

    print('starting training of rbm...')
    start_time = timeit.default_timer()

    for epoch in xrange(epochs):
        totalCost = 0.0
        for miniBatchIndex in xrange(nBatches):
            totalCost += train_model(miniBatchIndex) # aggregate miniBatch costs
        totalCost /= nBatches # average over miniBatches
        print('cost after epoch {0}: {1:.2f}'.format(epoch+1, totalCost))

    end_time = timeit.default_timer()
    run_time = '{0:.2f}'.format(end_time-start_time)

    opt_params = [param.get_value(borrow=False) for param in RBM.params]
    


    if nChannels>1:
        dimImage = int(np.sqrt(dimIn/nChannels)) # assume square image
        tile_shape = (10, nChannels) # plot first 10 filters across each channel
        X=opt_params[0].T.reshape(-1, dimImage*dimImage) # first nChannels rows relate to first filter
    else:
        dimImage = int(np.sqrt(dimIn)) # assume square image
        tile_shape = (10, 10) # or just plot first 100 filters
        X=opt_params[0].T
    
    image = Image.fromarray(
        tile_raster_images(
            X=X,
            img_shape=(dimImage, dimImage),
            tile_shape=tile_shape,
            tile_spacing=(1, 1)
        )
    )
    filtersFilename = 'rbm_filters_{0}_persistence={1}_k={2}.png'.format(dimHidden, persistent_bool, k) 
    image.save(filtersFilename)



    # Having trained the rbm, we can now sample from it, using gibbs_vhv. Use test
    # examples to initialise the chain (could have equally used training set)
    
    n_test = X_test.get_value(borrow=True).shape[0]

    # pick nChains random test examples with which to initialise nChains persistent chains
    # to run in parallel.
    
    test_idx = np_rng.randint(n_test - nChains)
    chain_init = theano.shared(
        np.asarray(
            X_test.get_value(borrow=True)[test_idx:test_idx + nChains],
            dtype=theano.config.floatX
        )
    )

    # define function to perform (parallel) vhv gibbs sampling using scan. This also
    # updates the state of the persistent chain with the new visible sample.
    # we can now afford to perform a large number of steps, to make sure the resulting samples
    # are a true reflection of the state of the model.
    
    nSteps = 1000 # steps before plotting
    
    ([hs_pre_sig, hs_mean, h_samples,
      vs_pre_sig, vs_mean, v_samples],
     updates) =  theano.scan(
         RBM.gibbs_vhv,
         outputs_info=[None, None, None, None, None, chain_init],
         n_steps=nSteps,
         name="gibbs_vhv")

    # append to updates to ensure update of the state of
    # our chain, so that the next nSteps iterations start from
    # the state at the end of the previous nSteps iterations.

    
    updates.update({chain_init: v_samples[-1]})
    

    # function to implement the persistent chain. vs_mean[-1] is used
    # to plot the samples - samples[-1] is only used to reinitialise
    # the state of persistent chain.

    sample_fn = theano.function(
        [],
        [vs_mean[-1], v_samples[-1]],
        updates=updates,
        name='sample_fn')
        

    # plot generated samples


    nChannelsPlot = min(nChannels, 3) # plot a maximum of 3 channels (intepreted as RGB)
    nullDims = tuple([None for i in range(4-nChannelsPlot)]) # tile_raster_images expects 4 element tuple - RGB and opacity (some of which can be None)
    

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well, hence dimImage+1 rather than dimImage)

    if nChannels>1:

        image_data = np.zeros(
            ((dimImage+1) * n_plotting_samples - 1, (dimImage+1) * nChains - 1, 4), # shape returned by tile_raster_images in the case of multiple channels (for compatability with Image package)
            dtype='uint8')
    else:

        image_data = np.zeros(
            ((dimImage+1) * n_plotting_samples - 1, (dimImage+1) * nChains - 1), # shape returned by tile_raster image otherwise
            dtype='uint8')
        
            
    for idx in range(n_plotting_samples):
        # This will plot n_plotting_samples samples
        # for each chain, which are separated by nSteps
        # intermediate samples in the gibbs chain. Note that
        # adjacent samples in the gibbs chain would not be worth
        # plotting as they are highly correlated.
        vs_mean, v_samples = sample_fn() 
        print('...plotting sample %d' % idx)

        if nChannels>1:
            X = vs_mean.reshape(nChains, nChannels, dimImage*dimImage).swapaxes(0,1)[:nChannelsPlot, :, :] 
            X = tuple(X) + nullDims

        else:
            X = vs_mean # nChains by dimImage*dimImage
        
        # 1 pixel left as space
        # Each row contains nChains samples that have been
        # sampled simultaneously. Adjacent rows
        # display samples separated by nSteps gibbs steps
        # note there will be a third returned dimension for multiple channels
        image_data[(dimImage+1) * idx:(dimImage+1) * idx + dimImage] = tile_raster_images(
            X=X,
            img_shape=(dimImage, dimImage),
            tile_shape=(1, nChains),
            tile_spacing=(1, 1)
        )

    # construct image

    image = Image.fromarray(image_data)
    samplesFilename = 'rbm_generated_samples_{0}_persistence={1}_k={2}.png'.format(dimHidden, persistent_bool, k)
    image.save(samplesFilename)


    # save all results

    trainingParams = {'epochs':epochs, 'miniBatchSize':miniBatchSize,
                   'alpha':alpha, 'dimHidden':dimHidden,
                   'persistent_bool': persistent_bool, 'k':k}

  
    with open('readme.txt', 'a') as f:
        f.write('\n {0}\n runtime: {1}\n {2}\n {3}\n'.format(trainingParams, run_time, filtersFilename, samplesFilename))

    f.close()

    os.chdir('../') # back to root folder


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

    X_train = train_set[0]
    X_train = toShared_x(X_train)
    X_test = test_set[0]
    X_test = toShared_x(X_test)

    nChannels=4
    dimHidden = 100
    epochs = 15
    miniBatchSize = 20
    alpha = 0.001
    nChains = 20
    n_plotting_samples = 10
    k = 15
    persistent_bool = True


    apply_rbm_sgd(X_train, X_test, nChannels, dimHidden, epochs, miniBatchSize, alpha, nChains,
                  n_plotting_samples, persistent_bool, k, results_dir)


if __name__ == "__main__":

    main(sys.argv[1:])



         
         
         
    

    

    
    

    
