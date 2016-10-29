import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def unit_scale(data):

    if type(data) is np.ndarray:
        data_max = np.max(data)
        data_min = np.min(data)
        data = (data-data_min)/(data_max-data_min).astype(np.float)
        
    elif type(data) is list:
        for i in range(len(data)):
            assert type(data[i]) is np.ndarray
            data_max = np.max(data[i])
            data_min = np.min(data[i])
            data[i] = (data[i]-data_min)/(data_max-data_min).astype(np.float)
    else:
        raise ValueError('incompatible type passed to unit_scale')
    
    return data


def pca(data, retain = False, monitoring=False):

    """ data should be [X_train, X_val, X_test] """
    
    # dimenisons should be number of samples by number of features (NxM)
    X_train, X_val, X_test = data 

    nTrain = X_train.shape[0]
    M = X_train.shape[1] # number of features


    X_train_feature_means =  np.mean(X_train, axis=0)[None, :]

    # zero mean the training data, and transform others in same way
    
    X_train -= X_train_feature_means
    X_val -= X_train_feature_means
    X_test -= X_train_feature_means
        
    cov = np.dot(X_train.T, X_train)/nTrain # M x M

    U, S, V = np.linalg.svd(cov) # cols of U hold P.Cs, M x M

    print('eigenvalues are: {}'.format(S)) # check eigenvalues
    
    # represent all datasets in new basis

    X_train = np.dot(X_train, U) 
    X_val = np.dot(X_val, U)
    X_test = np.dot(X_test, U)

    if monitoring:
        # check that covariance matrix in new basis is approx diagnonal

        cov_rot = np.dot(X_train.T, X_train)/nTrain # M by M

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cov_rot, interpolation='nearest')
        fig.colorbar(cax)
    
        plt.show()

    if retain:
        for i in range(len(S)):
            if (sum(S[:i])/sum(S) >= retain):
                break
            
            # i is now one past index of last retained P.C

        print('{} components have been retained'.format(i))
            
        # reduced data in new basis
        
        X_train = X_train[:, :i]
        X_val = X_val[:, :i]
        X_test = X_test[:, :i]

    
    return [X_train, X_val, X_test, U, S, X_train_feature_means] 

    
def zca(data, retain=False, monitoring=False):
    
    """ performs PCA before whitening result
    data should be [X_train, X_val, X_test]
    epsilon should be such that low pass filtering is achieved (usually 0.1 or 0.01)
    i.e. it should be larger than the smallest eigenvalues, which represent noise 
    """

    X_train, X_val, X_test, U, S, feature_means = pca(
        data, retain, monitoring)

    # whiten and regularise, applying same transformation to each dataset     

    k = X_train.shape[1] # the dimension of the PCA data (i.e. number of P.Cs retained)

    # Choose an epsilon appropriate for low pass filtering

    plt.figure()
    plt.plot([i for i in range(len(S[:k]))], S)
    plt.xlabel('component')
    plt.ylabel('eigenvalue')
    plt.title('eigenvalues corresponding to retained PCs')
    plt.show()
    
    while True:
        value = raw_input('choice for epsilon: ')
        try:
            epsilon = float(value)
            break
        except ValueError:
            print('Must be a number')

    # whiten
    X_train = (X_train/(np.sqrt(S[None, :k]+epsilon)))
    X_val = (X_val/(np.sqrt(S[None, :k]+epsilon)))
    X_test = (X_test/(np.sqrt(S[None, :k]+epsilon)))

    if monitoring:
        # check diagonality of X_train cov matrix post whitening
        cov = np.dot(np.transpose(X_train), X_train)/X_train.shape[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cov, interpolation='nearest')
        fig.colorbar(cax)
        plt.show()

    # rotate back to original basis

    X_train = np.dot(X_train, np.transpose(U))
    X_val = np.dot(X_val, np.transpose(U))
    X_test = np.dot(X_test, np.transpose(U))

    if monitoring:
        # check diagonality of X_train cov matrix again
        cov = np.dot(np.transpose(X_train), X_train)/X_train.shape[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cov, interpolation='nearest')
        fig.colorbar(cax)
        plt.show()


    return [X_train, X_val, X_test, U, S,
            feature_means, k, epsilon]





    









