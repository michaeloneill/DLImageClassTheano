import theano
import theano.tensor as T
import numpy as np
import timeit
import uuid

def supervised_tuning(params, train_model, validate_model, test_model,
                      nBatchTrain, nBatchVal, nBatchTest, sgd_opts,
                      monitoring_to_file=False):
    
    """ threshold is min number of mini-batch iterations """
    """ we keep track of model params so we can save it at optimum point """
    
    # set early stopping parameters
    
    threshold_increase = 2 # ratio by which threshold can be increased
    significance = 0.95 # relative improvement of this much deemed significant
    monitor_frequency = min(nBatchTrain, sgd_opts['monitor_frequency'])
        
    best_validation_error = np.inf
    best_iter = 0
    test_error = 0.0
    start_time = timeit.default_timer()
    stop = False
    epoch = 0

    max_epochs = sgd_opts['max_epochs']
    threshold = sgd_opts['min_epochs']*nBatchTrain
    alpha_init = sgd_opts['alpha_init']
    gamma = sgd_opts['gamma']
    p = sgd_opts['p']
    temp_monitoring_filename = None
    
    if monitoring_to_file:
        temp_monitoring_filename = str(uuid.uuid4())+'.txt'
        monitoring_file = open(temp_monitoring_filename, 'w')
    
    monitorCost = 0.0
    while (epoch < max_epochs) and (not stop):
        epoch = epoch +1
          
        for miniBatchIndex in xrange(nBatchTrain):
            iter = (epoch-1)*nBatchTrain+miniBatchIndex
            alpha = alpha_init/(1. + gamma*iter)**p # anneal learning rate

            # aggregate miniBatch costs
            monitorCost += train_model(miniBatchIndex, alpha)
            
            if (iter + 1) % monitor_frequency == 0:
                print('Training cost per minibatch for epoch {0}, minibatch {1}/{2} is : {3:.2f}'
                      .format(epoch, miniBatchIndex+1, nBatchTrain, monitorCost/(iter+1)))
                validation_error = np.mean(
                    [validate_model(i) for i in xrange(nBatchVal)]
                )
                print( 'Corresponding validation error is: {0:.2%}'
                       .format(validation_error)
                )
                
                if monitoring_to_file:
                    monitoring_file.write('{0} {1:.2f} {2:.2f}\n'.format(
                        iter+1, validation_error*100, monitorCost/(iter+1)))  

                if (validation_error < best_validation_error):
                    print('This is the best validation error so far')

                    if (validation_error < best_validation_error*significance):
                        threshold = max(threshold, iter*threshold_increase)                    

                    best_validation_error = validation_error
                    best_iter = iter+1
                        
                    test_error = np.mean(
                        [test_model(i) for i in range(nBatchTest)]
                    )

                    # borrow=False important here. It ensures best_params is guaranteed
                    # not to be aliased to shared variable params, which continues to change
                    # on subsequent loops. i.e best_params has its own space (down-side is that this
                    # is not memory efficient, because new space is created for it each time
                    # get_value() is called.
                    best_params = [param.get_value(borrow=False) for param in params] 
                    print('Corresponding test error is: {0:.2%}'
                          .format(test_error)
                    )
            if (iter >= threshold):
                stop = True
                break

    if monitoring_to_file:
        monitoring_file.close()
    
    end_time = timeit.default_timer()
    run_time = end_time-start_time

    print('Training complete. Best validation error of {0:.2%} obtained on iteration {2}'
          .format(best_validation_error, epoch, best_iter)
    )
    print('Corresponding test error of {0:.2%}'.format(test_error))
    print('Running time: {0:.2}s'.format(run_time))

    return [best_params, test_error, run_time, best_iter,
                temp_monitoring_filename]


def relu(z): return T.maximum(0.0, z)


def dropout_from_layer(input_dropout, p_dropout):

    """ Randomly deletes p_dropout  of input. """
    
    srng = theano.tensor.shared_randomstreams.RandomStreams(
                     np.random.RandomState(0).randint(999999))

    # 1s occur in mask where p>(1-p_dropout).
    mask = srng.binomial(n=1, p=1-p_dropout, size=input_dropout.shape)

    # Must perform cast, as int*float32 = float64,
    # which is incompatible with GPU.
    return input_dropout*T.cast(mask, theano.config.floatX)




