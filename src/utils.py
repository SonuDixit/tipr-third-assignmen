def load_mnist(path, kind='train'):
    """
    copied from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    """
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def unpickle_cifar(file):
    """
    for unpickling CIFAR data_batch, copied from http://www.cs.toronto.edu/~kriz/cifar.html
    usage:
    data_dict = unpickle_cifar(file)
    data = data_dict["data".encode('UTF-8')] # keys are ASCII chars, so str should be converted to UTF-8
                                              # https://stackoverflow.com/questions/6269765/what-does-the-b-character-do-in-front-of-a-string-literal
    labels = data_dict["labels".encode('UTF-8')] # list
    batch_label = data_dict["batch_label".encode('UTF-8')]
    :param file:
    :return:
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_batch_data_cifar(index):
    import os
    # from skimage.color import rgb2gray

    filename = os.path.join(os.path.dirname(os.getcwd()), "data", "CIFAR-10")
    file = os.path.join(filename, "data_batch_" + str(index))
    data_dict = unpickle_cifar(file)

    data = data_dict["data".encode('UTF-8')]
    labels = data_dict["labels".encode('UTF-8')]  # list
    data = data.reshape(data.shape[0],32,32,3)
    # data = rgb2gray(data)

    return data, labels

def get_batch_data_cifar_test():
    import os
    # from skimage.color import rgb2gray

    filename = os.path.join(os.path.dirname(os.getcwd()), "data", "CIFAR-10")
    file = os.path.join(filename, "test_batch")
    data_dict = unpickle_cifar(file)

    data = data_dict["data".encode('UTF-8')]
    labels = data_dict["labels".encode('UTF-8')]  # list
    data = data.reshape(data.shape[0],32,32,3)
    # data = rgb2gray(data)

    return data, labels

def tsne_transform_plot(data,label,n_components=2,fig_size=(10,10)):
    from sklearn import manifold
    from matplotlib import pyplot as plt
    perplexities = [5, 30, 50, 100]
    for p in perplexities:
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=p)
        y = tsne.fit_transform(data)
        plt.figure(figsize=fig_size)
        plt.title("perplexity " + str(p))
        plt.scatter(y[:, 0], y[:, 1], c=label, cmap="plasma")
        plt.colorbar()
        plt.show()
        plt.close()

def embedding_acc(true_label, pred_label):
    """
    :param true_label: list
    :param pred_label: list
    :return:
    """
    true_dict = {i:true_label.count(i) for i in range(10)}
    pred_dict = {i:pred_label.count(i) for i in range(10)}
    true_count = list(true_dict.values())
    true_count.sort(reverse=True)
    pred_count = list(pred_dict.values())
    pred_count.sort(reverse=True)
    acc = sum(min(true_count,pred_count))
    return acc*100/sum(pred_count)

