import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib import ticker
from matplotlib import cm
from PIL import Image
import math
import numpy as np
import sys, getopt

from DeepLearning.common.loadData import load_data_npz
from DeepLearning.common.preProcessing import unit_scale


def plot_confusion_matrix(confusion, label_descriptions):


        print(label_descriptions)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion, interpolation='nearest')
        fig.colorbar(cax)
        

        #show values in grid as well as colors

        nLabels = confusion.shape[0]
        assert nLabels == confusion.shape[1]
        
        for i in xrange(nLabels):
                for j in xrange(nLabels):
                       count = '{:.1g}'.format(confusion[i][j])
                       ax.text(i, j, count, va='center', ha='center', color='w', fontsize=10)
        
        ax.set_xticks(np.arange(len(label_descriptions)))
        ax.set_yticks(np.arange(len(label_descriptions)))

        ax.set_xticklabels(['']+label_descriptions, rotation=45, fontsize=10)
        ax.set_yticklabels(['']+label_descriptions, rotation=45, fontsize=10)

        # ax.set_xticklabels(['']+label_descriptions, fontsize=12)
        # ax.set_yticklabels(['']+label_descriptions, fontsize=12)

        
        ax.set_xlabel('Predictions', fontsize=12)
        ax.set_ylabel('Ground-Truth', fontsize=12)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.tight_layout()
        plt.show()


def plot_neuron_space_class_signatures(neuron_space_class_signatures,label_descriptions):

        nLabels,nNeurons = neuron_space_class_signatures.shape
        assert len(label_descriptions)==nLabels

        temp = np.ceil(nLabels/2)
        rows,cols = int(temp), int(np.ceil(nLabels/temp))
        
        
        fig, axes = plt.subplots(rows,cols, sharex='col', sharey='row')
        for i in xrange(rows):
                for j in xrange(cols):
                        axes[i][j].bar(
                                np.arange(1,nNeurons+1),neuron_space_class_signatures[i*cols+j],
                                width=0.4, color='b')
                        axes[i][j].set_title(label_descriptions[i*cols+j], fontsize=17)
                        axes[i][j].set_xlim(1,nNeurons)
                        axes[i][j].set_ylim(0,4.5)

        plt.xlim(1, nNeurons)
        plt.ylim(0, 4.5)
        big_ax = fig.add_subplot(1,1,1)
        big_ax.spines['top'].set_color('none')
        big_ax.spines['bottom'].set_color('none')
        big_ax.spines['left'].set_color('none')
        big_ax.spines['right'].set_color('none')
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none',top='off',bottom='off',left='off',right='off')
        big_ax.set_xlabel('neuron number', fontsize=18)
        big_ax.set_ylabel('mean activation', fontsize=18, labelpad=10)
        fig.tight_layout()
        plt.show()
        


def plot_bar(dist, xlabel, ylabel, label_descriptions):
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(dist)), dist, align='center', color='b', alpha=0.5)
        ax.set_xticks(np.arange(len(dist)))
        ax.set_xticklabels(label_descriptions, rotation=45, fontsize=12)
        #ax.set_xticklabels(label_descriptions, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        plt.tight_layout()
        plt.show()


def plot_labels_as_image(labels_arrays_to_plot, titles, nColors, label_descriptions, cmap=None, zoomParams=None):

        cmap = plt.cm.get_cmap(cmap, nColors)
        
        # main plot

        nPlots = len(labels_arrays_to_plot)
        assert nPlots==len(titles)
                
        fig, axes = plt.subplots(ncols=nPlots)

        # handle case when nPlots==1
        try:
                iter(axes)
        except TypeError:
                axes = [axes]

        for ax, array, title in zip(axes.flat, labels_arrays_to_plot, titles):
                im = ax.matshow(array,
                                interpolation='nearest',
                                cmap=cmap)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_title(title, fontsize=18)
        
                if zoomParams is not None:
                        axins = zoomed_inset_axes(ax,
                                                  zoomParams['zoom'],
                                                  loc=3)
                        axins.matshow(array,
                                      interpolation='nearest',
                                      cmap=cmap)
                        # remember origin is top left so x is vertical axis
                        axins.set_xlim(zoomParams['x1'], zoomParams['x2'])
                        axins.set_ylim(zoomParams['y2'], zoomParams['y1'])
                        plt.xticks(visible=False)
                        plt.yticks(visible=False)
                        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="1")

        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        cbaxis = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cbar = fig.colorbar(im, cax=cbaxis, ticks=range(nColors))
        cbar.ax.set_title('class labels', fontsize=18)
        im.set_clim(-0.5, nColors-0.5)
        cbar.ax.set_yticklabels(label_descriptions, rotation=45, fontsize=12)
        plt.show()


def plot_covariance(filename, labels, rastered=False):

        train_set, _, _ = load_data_npz(filename)
        X_train = train_set[0].astype(float)

        if rastered:
                X_train = train_set[0].swapaxes(3,1).reshape(-1,4)

        assert len(labels)==X_train.shape[1]
        
        cov = np.dot(np.transpose(X_train), X_train)/X_train.shape[0]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cov, interpolation='nearest')
        fig.colorbar(cax)

        ax.set_xticklabels(['']+labels)
        ax.set_yticklabels(['']+labels)

        plt.show()
        
        
def plot_spec_profile(filename, class_label, channel_descriptions, zorders, colors, scale_to_unit_interval=False, rastered_input=False):

        """ plots spectral profile for class class_label """
        
        train_set, _, _ = load_data_npz(filename)
        X_train = train_set[0]
        y_train = train_set[1]
        assert len(zorders)==len(colors)
        
        if scale_to_unit_interval:
                X_train = unit_scale(X_train)
        if rastered_input:
                # X_train must be of shape [nSamples, nChannels, rows, cols]
                assert len(X_train.shape)==4
                spectral_intensities = np.squeeze(np.apply_over_axes(np.mean, X_train, [2,3])) # average over raster
                xlabel = 'Average channel intensity (normalised)'

        else:
                spectral_intensities = X_train
                xlabel = 'Channel intensity (normalised)'
                
        
        plt.figure()
        for channel in xrange(spectral_intensities.shape[1]):
                plt.hist(spectral_intensities[y_train==class_label, channel], bins=100,
                         label=channel_descriptions[channel],
                         zorder=zorders[channel], color=colors[channel])
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel('number of samples', fontsize=14)
        plt.legend()
        plt.show()


def plot_channel_hists(filename, channel, class_labels, class_label_descriptions, zorders, colors, scale_to_unit_interval=False, rastered_input=False):

        """ plots histograms for each class in class_labels for channel channel """
        
        train_set, _, _ = load_data_npz(filename)
        X_train = train_set[0]
        y_train = train_set[1]
        assert len(class_labels)==len(class_label_descriptions)==len(zorders)==len(colors)
        assert max(class_labels)<=y_train.max() and min(class_labels)>=y_train.min()
        
        if scale_to_unit_interval:
                X_train = unit_scale(X_train)
        if rastered_input:
                # X_train must be of shape [nSamples, nChannels, rows, cols]
                assert len(X_train.shape)==4
                spectral_intensities = np.squeeze(np.apply_over_axes(np.mean, X_train, [2,3])) # average over raster
                xlabel = 'Average channel intensity (normalised)'

        else:
                spectral_intensities = X_train
                xlabel = 'Channel intensity (normalised)'
                
        
        plt.figure()
        for label, description, zorder, color in zip(class_labels,
                                                     class_label_descriptions,
                                                     zorders, colors):
                plt.hist(spectral_intensities[y_train==label, channel], bins=100,
                         label=description,
                         zorder=zorder, color=color)
        plt.xlim(0.0,1.0)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel('number of samples', fontsize=14)
        plt.legend()
        plt.show()



def concatenate_vert(images, deltaW, offsetW, offsetH):

        """ concatenates 2 images vertically """
        
        images = map(Image.open, images)
        W = max(img.size[0] for img in images)
        H = sum(img.size[1] for img in images)

        result = Image.new("RGBA", (W, H))

        result.paste(images[0], (0, 0))
        
        # re-sizing 
        new_width = images[0].size[0]-deltaW
        ratio = new_width/float(images[1].size[0])
        new_height = int(images[1].size[1]*ratio)
        
        img = images[1].resize((new_width, new_height), Image.ANTIALIAS)
        result.paste(img, (offsetW, images[0].size[1]-offsetH))
        result.save('result.png')
                    

def plot_predictions(preds, errors, labels, colors, xtick_labels, x_label):

    ind = np.arange(preds.shape[0])
    width = 0.4
    
    fig, ax = plt.subplots()
    pos = ind
    for col in xrange(preds.shape[1]):
        ax.bar(pos, preds[:,col], width,
               color=colors[col],
               alpha=0.4,
               yerr=errors[:, col],
               error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2),
               label=labels[col])
        pos = pos+width
        
    ax.set_xticks(ind+width)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylim(0,1)
    ax.set_ylabel('Probability', fontsize=13, labelpad=20)
    ax.legend()
    plt.show()


def plot_validation_curves(files, labels, colors, linestyles):

    fig, ax = plt.subplots()
    
    for f, l, c, s in zip(files, labels, colors, linestyles):
            file = open(f, 'r')
            data = np.loadtxt(file)
            file.close()
            ax.plot(data[:,0], data[:,1], label=l, color=c, linestyle=s)

    ax.set_xlim(0,100000)
    ax.set_ylim(3.5, 10)
    ax.set_xlabel('mini-batch iterations', fontsize=14)
    ax.set_ylabel('validation set error (%)', fontsize=14)

    plt.legend(loc=1, fontsize=16)
    
    # axins = zoomed_inset_axes(ax,
    #                           2.5,
    #                           loc=1)
    
    # for f, l, c, s in zip(files, labels, colors, linestyles):
    #         file = open(f, 'r')
    #         data = np.loadtxt(file)
    #         file.close()
    #         axins.plot(data[:,0], data[:,1], label=l, color=c, linestyle=s)


    # axins.set_xlim(900000,1000000)
    # axins.set_ylim(0.5,1.5)

    # # remember origin is top left so x is vertical axis
    # plt.xticks(visible=False)
    # #plt.yticks(visible=False)
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.7")

    plt.show()
    

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar = ndar*1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.
    
    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices used for transforming those rows
    (such as the first layer of a neural net).
    
    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats. Should only be used if values have been scaled to
    unit interval, either with scale_rows_to_unit_interval or before passing.

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """
    
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2
    
    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]
    
    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        # colors default to 0 (i.e. black), alphas defaults to 1 (fully opaque i.e.
        # corresponding pixel fully visible in image))
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8') 
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype) 

        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]
                         
        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array
                        
    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing
        
        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.ones(out_shape, dtype=dt)*255
            
        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array
                                            
                                            
def plotPatches(X, title):
    # X is patchDim*patchDIm by numPatches
    # Columns are reshaped and plotted as square images

    numPatches = X.shape[1]
    patchDim = math.sqrt(X.shape[0])
    assert (patchDim*patchDim == X.shape[0]), "Patches not square!"

    patches = X.reshape((patchDim, patchDim, numPatches))
    gridRows = math.ceil(math.sqrt(numPatches))
    gridCols = math.ceil(numPatches/gridRows)
    
    fig = plt.figure(title)
    for row in range(int(gridRows)):
        lastCol = gridCols if ((row*gridCols+gridCols)<=numPatches) else (numPatches-row*gridCols)
        for col in range(int(lastCol)):
            ax = fig.add_subplot(gridRows, gridCols, gridCols*row+col+1)
            ax.matshow(patches[:, :, gridCols*row+col], cmap=cm.binary)
            plt.xticks([])
            plt.yticks([])
    plt.suptitle(title)
    plt.show()
        

def plotMatrix(X, title):
    fig = plt.figure(title)
    plt.imshow(X)
    plt.title(title)
    plt.show()



def main(argv):

    try:
        opts, args = getopt.getopt(argv, 'hf:l:c:s:', ['--file=', '--label=', '--color=', '--style='])
    except getopt.GetoptError:
        print 'incorrect usage'
        print 'usage: plotting.py -f <file> -l <label> -c <color> -s <style>'

    files = []
    labels = []
    colors= []
    linestyles = []
    
    for opt, arg in opts:
        if opt == '-h':
            print 'usage: plotting.py -f <file> -l <label> -c <color> -s <style>'
        elif opt in('-f', '--file'):
            files.append(arg)
        elif opt in ('-l', '--label'):
            labels.append(arg)
        elif opt in ('-c', '--color'):
            colors.append(arg)
        elif opt in ('-s', '--style'):
            assert arg=='-' or arg=='--', 'invalid linestyle argument'
            linestyles.append(arg)
        
    if (len(files)==0 or len(labels)==0):
        print 'incorrect usage'
        print 'usage: plotting.py -f <file> -l <label> -c <color> -s <style>'

    else:
        assert len(files) == len(labels) and len(files) == len(linestyles) and len(files) == len(colors), 'Number of files, labels, colors and styles do not match'
        plot_validation_curves(files, labels, colors, linestyles)
        

if __name__ == '__main__':
        main(sys.argv[1:])

        # concatenate_vert(['mc_dropout_predictions_mnist_rot_1_100.png', 'mnist_1_rotated_10.png'],
        #                  deltaW=178, offsetW=100, offsetH=59)
    

        # histogram for each seperate class (one channel) for sat4

        # plot_channel_hists('./datasets/sat4-sat6/sat4_scaled_ZCA_retain_False_epsilon_0.001_rastered.npz',
        #                   zorders=[2,3,4,1],
        #                   class_labels = [0,1,2,3],
        #                   class_label_descriptions = ['barren', 'trees', 'grassland', 'other'],
        #                   colors=['#996600', '#003300', '#339933', '#cc9900'],
        #                   channel=0,  
        #                   scale_to_unit_interval=True,
        #                   rastered_input=True)

        
        # histogram for each of selected classes (one channel) for landsat2
        
        # plot_channel_hists('./datasets/KisanHub/landsat2/landsat2.npz',
        #                    zorders=[0,1,2,3],
        #                    class_labels = [0,1,2,3],
        #                    class_label_descriptions = ['a','b','c','d'],
        #                    colors=['#996600', '#003300', '#339933', '#cc9900'],
        #                    channel=0,
        #                    scale_to_unit_interval=True)

        
        
        # labels = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
        # plot_covariance('./datasets/KisanHub/landsat1.npz', labels)

        # plot_spec_profile('./datasets/sat4-sat6/sat4_scaled_ZCA_retain_False_epsilon_0.001_rastered.npz',
        #                   class_label=1,
        #                   channel_descriptions=['B','G','R', 'NIR'], 
        #                   zorders=[0,1,2,3],
        #                   colors=['b', 'g', 'r', 'm'],
        #                   scale_to_unit_interval=True,
        #                   rastered_input=True)
                          




        

