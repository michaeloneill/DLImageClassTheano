import cPickle
import scipy.io
import numpy as np
import sys, getopt
import scipy.ndimage as nd

import matplotlib.pyplot as plt
from PIL import Image
from DeepLearning.common.plotting import tile_raster_images
from DeepLearning.common.loadData import load_data_pickled, load_data_npz
from DeepLearning.common.preProcessing import unit_scale, pca, zca

from osgeo import gdal


def sat4_mc_test(label, num_corruptions, channel):

    """ Generates a dataset from single data sample corrupted multiple times."""

    inFilename = './datasets/sat4-sat6/sat4_scaled_ZCA_retain_False_epsilon_0.001_rastered.npz'
    train_set, _, _ = load_data_npz(inFilename)

    # take first example of given label
    sample = train_set[0][train_set[1]==label][0]
    print 'taking sample shape as:{}'.format(sample.shape)

    corruptions = np.zeros((num_corruptions+1, sample.shape[0], sample.shape[1], sample.shape[2]))
    scaling = 1.0
    for i in xrange(num_corruptions+1):
        print scaling
        temp = sample
        temp[channel] = temp[channel]*scaling
        corruptions[i] = temp
        scaling -= 0.1

    print 'dataset shape is:{}'.format(corruptions.shape)
    outFilename = './datasets/sat4-sat6/sat4_label_{0}_corrupted_{1}_channel_{2}.npz'.format(label, num_corruptions, channel)
    with open(outFilename, 'wb') as f:
        np.savez(f, X_test=corruptions)
    f.close()
        


def generate_testset(X_filename, outFilename, y_filename=None, transforms_filename=None):

    X_test = unit_scale(np.array(gdal.Open(X_filename).ReadAsArray()))

    X_test = X_test.transpose((1,2,0))
    image = Image.fromarray((X_test[:,:,[2, 1, 0]]*255).astype('uint8')) # flip to RGB
    image.show()

    imageShape = X_test.shape[:2]
    print 'image shape is {}'.format(imageShape)
    nChannels = X_test.shape[2]
    X_test = X_test.reshape(-1, nChannels)

    if transforms_filename is not None:
        
        with np.load(transforms_filename) as transforms:
            U = transforms['U']
            print 'Taking U as {}'.format(U)
            S = transforms['S']
            print 'Taking S as {}'.format(S)
            feature_means = transforms['feature_means']
            print 'Taking feature means as {}'.format(feature_means)
            nRet = transforms['nRet']
            print 'Taking nRet as {}'.format(nRet)
            epsilon = transforms['epsilon']
            print 'Taking epsilon as {}'.format(epsilon)

        k = X_test.shape[1]
        assert feature_means.shape[1]==k
        X_test = X_test - feature_means # center
        X_test = np.dot(X_test, U) # rotate onto PCs
        if nRet is not None:
            X_test = X_test[:, :nRet]
            k = nRet

        X_test = (X_test/(np.sqrt(S[None, :k]+epsilon))) # whiten
        X_test = np.dot(X_test, np.transpose(U)) # rotate back

    if y_filename is not None:
        y_test = np.array(gdal.Open(y_filename).ReadAsArray()).astype('uint8')
        assert imageShape == y_test.shape
        y_test = y_test.ravel()-1 # index from zero
        print 'y_max is {}'.format(y_test.max())
        print 'y_min is {}'.format(y_test.min())
    else:
        y_test = None

    with open(outFilename, 'wb') as f:
        np.savez(f, X_test=X_test, y_test=y_test)
    f.close()


def from_tiff_ZCA(dataFiles, labelFiles, lim_train, outFilePrefix, retain=False, monitoring=False):

    train_set, validation_set, test_set = from_tiff(dataFiles, labelFiles, lim_train)
    
    if monitoring:
        # check diagonality of X_train cov pre ZCA
        cov = np.dot(np.transpose(X_train), X_train)/X_train.shape[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cov, interpolation='nearest')
        fig.colorbar(cax)
        plt.show()

    X_train, X_val, X_test, U, S, feature_means, k, epsilon = zca(
        [train_set[0], validation_set[0], test_set[0]], retain, monitoring)

    dataset_info_filename = outFilePrefix + '_retain_{0}_epsilon_{1}.txt'.format(retain, epsilon)
    with open(dataset_info_filename, 'w') as f:
        f.write('U is {0}\n S is {1}\n feature_means are {2}\n nRet is {3}\n epsilon is {4}\n'
                .format(U, S, feature_means, k, epsilon))

    datasetFilename = outFilePrefix + '_retain_{0}_epsilon_{1}.npz'.format(retain, epsilon)

    with open(datasetFilename, 'wb') as f:
        np.savez(f, X_train=X_train,
                 y_train=train_set[1],
                 X_val=X_val,
                 y_val=validation_set[1],
                 X_test=X_test,
                 y_test=test_set[1])
    f.close()

    transformsFilename = outFilePrefix + '_retain_{0}_epsilon_{1}_transforms.npz'.format(retain, epsilon)
    with open(transformsFilename, 'wb') as f:
        np.savez(f, U=U, S=S, feature_means=feature_means, nRet=k, epsilon=epsilon) 
    f.close()


def from_tiff(dataFiles, labelFiles, lim_train, dataFilename=None):

    data = []
    labels = []

    for dataFile, labelFile in zip(dataFiles, labelFiles):
        data.append(unit_scale(np.array(gdal.Open(dataFile).ReadAsArray())))
        labels.append(np.array(gdal.Open(labelFile).ReadAsArray()).astype('uint8'))

    for i in xrange(len(data)):
        data[i] = data[i].transpose((1, 2, 0))
        assert data[i][:,:,0].shape == labels[i].shape
        image = Image.fromarray((data[i][:,:,[2, 1, 0]]*255).astype('uint8')) # flip to RGB
        image.show()

    nChannels = data[0].shape[2]
    for i in xrange(1, len(data)):
        assert data[i].shape[2] == nChannels

    X = np.concatenate(tuple(d.reshape(-1, nChannels) for d in data))
    y = np.concatenate(tuple(l.ravel()-1 for l in labels))
    print 'X shape {}'.format(X.shape)
    print 'X max {}'.format(data[i].max())
    print 'X min {}'.format(data[i].min())

    print 'y shape {}'.format(y.shape)
    print 'y max {}'.format(y.max())
    print 'y min {}'.format(y.min())

    # split into train,val,test

    lim_val = lim_train+(1.0-lim_train)/2
    
    nSamples = X.shape[0]
    indices = np.random.permutation(nSamples)
    
    X_train = X[indices[:lim_train*nSamples]]
    y_train = y[indices[:lim_train*nSamples]]
    print(y_train[:50])
    
    X_val = X[indices[lim_train*nSamples:lim_val*nSamples]]
    y_val = y[indices[lim_train*nSamples:lim_val*nSamples]]
    
    X_test = X[indices[lim_val*nSamples:]]
    y_test = y[indices[lim_val*nSamples:]]


    print 'X_train shape {}'.format(X_train.shape)
    print 'y_train shape {}'.format(y_train.shape)
    print 'X_train max {}'.format(X_train.max())
    print 'X_train min {}'.format(X_train.min())
    print 'y_train max {}'.format(y_train.max())
    print 'y_train min {}'.format(y_train.min())


    print 'X_val shape {}'.format(X_val.shape)
    print 'y_val shape {}'.format(y_val.shape)
    print 'X_val max {}'.format(X_val.max())
    print 'X_val min {}'.format(X_val.min())
    print 'y_val max {}'.format(y_val.max())
    print 'y_val min {}'.format(y_val.min())


    print 'X_test shape {}'.format(X_test.shape)
    print 'y_test shape {}'.format(y_test.shape)
    print 'X_test max {}'.format(X_test.max())
    print 'X_test min {}'.format(X_test.min())
    print 'y_test max {}'.format(y_test.max())
    print 'y_test min {}'.format(y_test.min())

    
               
    if dataFilename is not None:
        
        with open(dataFilename, 'wb') as f:
            np.savez(f, X_train=X_train, y_train=y_train, X_val=X_val,
                     y_val=y_val, X_test=X_test, y_test=y_test)
        f.close()

    else:

        train_set = [X_train, y_train]
        validation_set = [X_val, y_val]
        test_set = [X_test, y_test]
        
        return [train_set, validation_set, test_set]


def landsat1(dataFiles, to_file=False):
        
    train_x = np.array(gdal.Open(dataFiles['train_x']).ReadAsArray())
    val_x = np.array(gdal.Open(dataFiles['val_x']).ReadAsArray())
    test_x = np.array(gdal.Open(dataFiles['test_x']).ReadAsArray())
    labels = np.array(gdal.Open(dataFiles['labels']).ReadAsArray())

    
    nChannels = train_x.shape[0]
    assert nChannels==val_x.shape[0]==test_x.shape[0]
    
    X_train = np.zeros((train_x.shape[1], train_x.shape[2], train_x.shape[0]), 'uint8')
    X_val = np.zeros((val_x.shape[1], val_x.shape[2], val_x.shape[0]), 'uint8')
    X_test = np.zeros((test_x.shape[1], val_x.shape[2], test_x.shape[0]), 'uint8')
    y_train = np.zeros((labels.shape[0], labels.shape[1]), 'uint8')

    for i in xrange(nChannels):
        X_train[:,:,i] = train_x[i]*255
        X_val[:,:,i] = val_x[i]*255
        X_test[:,:,i] = test_x[i]*255

    y_train = labels
        
    # image = Image.fromarray(X_train[:,:, :3])    
    # image.show()

    X_train = X_train.reshape((-1, nChannels))
    X_val = X_val.reshape((-1, nChannels))
    X_test = X_test.reshape((-1, nChannels))
    y_train = y_train.ravel()

    if to_file:
        
        dataFilename = './datasets/KisanHub/landsat1.npz'
        with open(dataFilename, 'wb') as f:
            np.savez(f, X_train=X_train, y_train=y_train, X_val=X_val,
                     y_val=y_train, X_test=X_test, y_test=y_train)
        f.close()

    else:

        train_set = [X_train, y_train]
        validation_set = [X_val, y_train]
        test_set = [X_test, y_train]
        
        return [train_set, validation_set, test_set]


def mnist_rotations(digit, numRots):

    """ Generates dataset from single mnist digit rotated between 0 and 180 degrees.
    Returned data is of shape numRots by 28*28
    """
    
    inFilename = './datasets/mnist/mnist.pkl'
    train_set, _, _ = load_data_pickled(inFilename)

    # Take the first example of digit from mnist.
    sample = train_set[0][train_set[1]==digit][0].reshape((28,28))

    X_rot = np.zeros((numRots, 28*28))
    angle = 0
    d_angle = 180/numRots
    for index in xrange(numRots):
        angle += d_angle
        X_rot[index] = nd.interpolation.rotate(sample, angle, reshape=False).ravel()

    imageFilename = 'mnist_{0}_rotated_{1}.png'.format(digit, numRots)
    image = Image.fromarray(tile_raster_images(
        X=X_rot,
        img_shape=(28, 28), tile_shape=(1,numRots),
        tile_spacing=(1, 1)))
    image.save(imageFilename)

    X_rot = np.reshape(X_rot, (-1, 1, 28, 28)) 

    dataFilename = './datasets/mnist/mnist_{0}_rotated_{1}.pkl'.format(digit, numRots)
    with open(dataFilename, 'wb') as f:
        cPickle.dump(X_rot, f, -1)
    f.close()


def sat4_ZCA(retain=False, unroll=False):

    """ ZCA is performed on the set of RGBNIR pixel values 
    throughout the sat4 dataset."""

    train_set, validation_set, test_set = sat4_rastered()

    # plot first sample in first channel pre ZCA

    plt.figure()
    plt.imshow(train_set[0][0, 0, :, :])
    plt.show()
    
    # rows become pixels (unrolled transpose of original data matrices)
    # and cols the corresponding channels.
    X_train = train_set[0].swapaxes(3,1).reshape(-1,4)
    X_val = validation_set[0].swapaxes(3,1).reshape(-1,4)
    X_test = test_set[0].swapaxes(3,1).reshape(-1,4)
    
    X_train_ZCA, X_val_ZCA, X_test_ZCA, _, _, _, _, epsilon = zca([X_train, X_val, X_test], retain)


    k = X_train_ZCA.shape[1] # number of components retained
    
    # undo transformation pre ZCA

    X_train_ZCA = X_train_ZCA.reshape(-1, 28, 28, k).swapaxes(3,1)
    X_val_ZCA = X_val_ZCA.reshape(-1, 28, 28, k).swapaxes(3,1)
    X_test_ZCA = X_test_ZCA.reshape(-1, 28, 28, k).swapaxes(3,1) 

    # plot first sample in first channel for sanity check

    plt.figure()
    plt.imshow(X_train_ZCA[0, 0, :, :])
    plt.show()
    
    if unroll:

        X_train_ZCA = X_train_ZCA.reshape(-1, k*28*28)
        X_val_ZCA = X_val_ZCA.reshape(-1, k*28*28)
        X_test_ZCA = X_test_ZCA.reshape(-1, k*28*28)

        data = [X_train_ZCA, X_val_ZCA, X_test_ZCA]
        
        X_train_ZCA, X_val_ZCA, X_test_ZCA = unit_scale(data) # scale again for compatability with sdA and dbn

        print(np.max(X_train_ZCA))
        print(np.min(X_train_ZCA))

        print(np.max(X_val_ZCA))
        print(np.min(X_val_ZCA))

        print(np.max(X_test_ZCA))
        print(np.min(X_test_ZCA))
        
        outFilename = './datasets/sat4-sat6/sat4_scaled_ZCA_retain_{0}_epsilon_{1}_unrolled.npz'.format(retain, epsilon)

    else:

        # use with cnn
        # no need to re-scale
        
        outFilename = './datasets/sat4-sat6/sat4_scaled_ZCA_retain_{0}_epsilon_{1}_rastered.npz'.format(retain, epsilon)

    with open(outFilename, 'wb') as f:
        np.savez(f, X_train=X_train_ZCA, y_train=train_set[1], X_val=X_val_ZCA, y_val=validation_set[1], X_test=X_test_ZCA, y_test=test_set[1])
    f.close()


def mnist_rastered():

    inFilename = './datasets/mnist/mnist.pkl'
    train_set, validation_set, test_set = load_data_pickled(inFilename)
    
    # image = Image.fromarray(tile_raster_images(
    #     X=train_set[0],
    #     img_shape=(28, 28), tile_shape=(10,10),
    #     tile_spacing=(1, 1)))
    # image.save('mnist_digits.png')


    X_train = np.reshape(train_set[0], (-1, 1, 28, 28)) 
    X_val = np.reshape(validation_set[0], (-1, 1, 28, 28))
    X_test = np.reshape(test_set[0], (-1, 1, 28, 28))
    
    # leave y unchanged
    train_set = [X_train, train_set[1]]
    validation_set = [X_val, validation_set[1]]
    test_set = [X_test, test_set[1]]

    data = [train_set, validation_set, test_set]

    outFilename = './datasets/mnist/mnist_rastered.pkl'
    with open(outFilename, 'wb') as f:
        cPickle.dump(data, f, -1)
    f.close()



def sat4_rastered(to_file=False):

    inFilename = './datasets/sat4-sat6/sat-4-full.mat'
    data = scipy.io.loadmat(inFilename)

    # number of samples by no. input feature maps by height by width
    # also rescaling to [0,1]
    X_train = np.transpose(data['train_x'], (3, 2, 0, 1))/float(255) 
    y_train = np.argmax(data['train_y'], axis=0).T
    X_val = np.transpose(data['test_x'][:, :, :, :50000], (3, 2, 0, 1))/float(255)
    y_val = np.argmax(data['test_y'][:, :50000], axis=0).T
    X_test = np.transpose(data['test_x'][:, :, :, 50000:], (3, 2, 0, 1))/float(255)
    y_test = np.argmax(data['test_y'][:, 50000:], axis=0).T

    # # plot training images in colour
    
    # X = tuple(X_train.swapaxes(0,1).reshape(4, -1, 28*28)[:3, :, :]) + tuple([None])
    # image = Image.fromarray(
    #     tile_raster_images(
    #         X=X,
    #         img_shape=(28, 28),
    #         tile_shape=(10, 10),
    #         tile_spacing=(1, 1)
    #     )
    # )
    # image.save('sat_images.png')

    
    if to_file:

        outFilename = './datasets/sat4-sat6/sat4_scaled_rastered.npz'
        with open(outFilename, 'wb') as f:
            np.savez(f, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
        f.close()

    else:

        train_set = [X_train, y_train]
        validation_set = [X_val, y_val]
        test_set = [X_test, y_test]
        
        return [train_set, validation_set, test_set]


    
def sat4_unroll(to_file=False):

    inFilename = './datasets/sat4-sat6/sat-4-full.mat'
    data = scipy.io.loadmat(inFilename)

    # note rescaling to [0,1]
    X_train = data['train_x'].reshape((28*28*4, -1)).T/float(255) 
    y_train = np.argmax(data['train_y'], axis=0).T
    X_val = data['test_x'][:, :, :, :50000].reshape((28*28*4, -1)).T/float(255)
    y_val = np.argmax(data['test_y'][:, :50000], axis=0).T
    X_test = data['test_x'][:, :, :, 50000:].reshape((28*28*4, -1)).T/float(255)
    y_test = np.argmax(data['test_y'][:, 50000:], axis=0).T
    
    
    if to_file:

        outFilename = './datasets/sat4-sat6/sat4_scaled_unrolled.npz'
        with open(outFilename, 'wb') as f:
            np.savez(f, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
        f.close()

    else:

        train_set = [X_train, y_train]
        validation_set = [X_val, y_val]
        test_set = [X_test, y_test]
        
        return [train_set, validation_set, test_set]


if __name__=='__main__':

    print 'generating dataset...'
    sat4_rastered(to_file=True)
    #sat4_unroll(to_file=True)
    #sat4_ZCA() 



    
    #mnist_rastered()
    #sat4_ZCA() # rastered 
    #sat4_ZCA(unroll=True)
    #mnist_rotations(1, 10)
    #sat4_mc_test(label=2, num_corruptions=10, channel=2)


    # for landsat 1

    # dataFiles = {'train_x':'./datasets/KisanHub/landsat1/Clip_RT_LC8_201_024_2016_111.tif',
    #              'val_x':'./datasets/KisanHub/landsat1/Clip_RT_LC8_201_024_2016_015.tif',
    #              'test_x': './datasets/KisanHub/landsat1/Clip_RT_LC8_201_024_2016_047.tif',
    #              'labels':'./datasets/KisanHub/landsat1/Classify_RT_LC8_201_024_2016_111.tif'}
    
    #landsat1(dataFiles)
    # landsat_ZCA(dataFiles,'./datasets/KisanHub/landsat1/landsat1_ZCA')







    # for landsat 2

    # dataFiles = ['./datasets/KisanHub/landsat2/Raw_Data/RT_LC8_201_024_2015_108.tif',
    #              './datasets/KisanHub/landsat2/Raw_Data/RT_LC8_201_024_2015_220.tif',
    #              './datasets/KisanHub/landsat2/Raw_Data/RT_LC8_201_024_2016_015.tif',
    #              './datasets/KisanHub/landsat2/Raw_Data/RT_LC8_201_024_2016_047.tif',
    #              './datasets/KisanHub/landsat2/Raw_Data/RT_LC8_201_024_2016_111.tif']

    # labelFiles = ['./datasets/KisanHub/landsat2/Classified/Classify_003_RT_LC8_201_024_2015_108.tif',
    #              './datasets/KisanHub/landsat2/Classified/Classify_003_RT_LC8_201_024_2015_220.tif',
    #              './datasets/KisanHub/landsat2/Classified/Classify_003_RT_LC8_201_024_2016_015.tif',
    #              './datasets/KisanHub/landsat2/Classified/Classify_003_RT_LC8_201_024_2016_047.tif',
    #              './datasets/KisanHub/landsat2/Classified/Classify_003_RT_LC8_201_024_2016_111.tif']

    #from_tiff(dataFiles, labelFiles, lim_train=0.9998, dataFilename = './datasets/KisanHub/landsat2/landsat2.npz')
    #from_tiff_ZCA(dataFiles, labelFiles, lim_train=0.9998, './datasets/KisanHub/landsat2/landsat2_ZCA')
    # generate_testset(
    #     X_filename = './datasets/KisanHub/landsat2/Raw_Data/RT_LC8_201_024_2016_111.tif',
    #     y_filename = './datasets/KisanHub/landsat2/Classified/Classify_003_RT_LC8_201_024_2016_111.tif',
    #     outFilename = './datasets/KisanHub/landsat2/2016_111_transformed_testset.npz',
    #     transforms_filename = './datasets/KisanHub/landsat2/landsat2_ZCA_retain_False_epsilon_1e-05_transforms.npz')






    # for RapidEye

    
    # dataFiles = ['./datasets/KisanHub/RE/Raw_Data/3063223_2016-05-04_RE4_3A_380026.tif',
    #              './datasets/KisanHub/RE/Raw_Data/3063223_2016-05-08_RE3_3A_381496.tif']

    # labelFiles = ['./datasets/KisanHub/RE/Classified_pix/RE_Class_6_2016_05_04.tif',
    #               './datasets/KisanHub/RE/Classified_pix/RE_Class_6_2016_05_08.tif']


    #from_tiff(dataFiles, labelFiles, 0.998, dataFilename='./datasets/KisanHub/RE/RE_pix.npz')
    #from_tiff_ZCA(dataFiles, labelFiles, 0.998, './datasets/KisanHub/RE/RE_pix_ZCA')
    
    # generate_testset(
    #     X_filename = './datasets/KisanHub/RE/Raw_Data/3063223_2016-05-04_RE4_3A_380026.tif',
    #     y_filename = './datasets/KisanHub/RE/Classified_pix/RE_Class_6_2016_05_04.tif',
    #     outFilename = './datasets/KisanHub/RE/2016_05_04_transformed_testset.npz',
    #     transforms_filename = './datasets/KisanHub/RE//RE_pix_ZCA_retain_False_epsilon_0.0001_transforms.npz')


    
    

    
    
